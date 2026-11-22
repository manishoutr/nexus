"""Ingestion jobs: claim, process, retry, recover.

The job table is the source of truth. Whatever triggers a run (the threaded
poller or a Celery task), the same run_job() does the work, so the safety
properties hold either way:

* Atomic claim: a conditional UPDATE (status pending -> processing) succeeds
  for exactly one caller, so two workers never process the same job.
* Backoff: a failed attempt sets next_run_at = now + base * 2^(attempt-1).
* Permanent errors (corrupt file, missing document) fail at once; only
  unexpected errors are retried.
* Recovery: a job stuck in "processing" past JOB_STALE_AFTER_S (worker died)
  is put back in the queue, or failed if it has used all attempts.
* Idempotent: chunks for a document are replaced in one transaction, so a
  retry after a crash never leaves duplicates.
"""
import json
import logging
import os
import socket
import threading
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import update

from ..config import settings
from ..db import SessionLocal
from ..models import Chunk, Document, IngestionJob
from .chunk import chunk_document
from .embeddings import backend_name, embed_many
from .extract import extract_pages
from . import vectorstore
from .storage import get_storage
from .validate import UploadRejected, validate_upload

log = logging.getLogger("nexus.worker")
_stop = threading.Event()
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def jlog(event: str, **fields) -> None:
    """One JSON object per log line, so logs can be grepped and parsed."""
    log.info(json.dumps({"event": event, **fields}, default=str))


class PermanentError(Exception):
    """Retrying cannot help (bad file, document deleted)."""


def _now() -> datetime:
    return datetime.utcnow()


# --- processing -------------------------------------------------------------

def process_document(db, document_id: int) -> int:
    """Extract, chunk, embed and store one document. Returns chunk count."""
    doc = db.query(Document).filter_by(id=document_id).first()
    if doc is None:
        raise PermanentError(f"document {document_id} no longer exists")
    storage = get_storage()
    key = doc.storage_key or f"{doc.workspace_id}/{doc.filename}"  # pre-UUID rows
    try:
        data = storage.get(key)
    except FileNotFoundError:
        raise PermanentError("stored file is missing")
    try:
        validate_upload(doc.filename, data, settings.max_upload_mb * 1024 * 1024)
        pages = extract_pages(data, doc.filename)
    except UploadRejected as e:
        raise PermanentError(e.message)
    chunks = chunk_document(pages)
    if not chunks:
        raise PermanentError("no text could be extracted (scanned or image-only file?)")

    vectors = embed_many([c["text"] for c in chunks])
    model = backend_name()
    db.query(Chunk).filter_by(document_id=doc.id).delete()  # idempotent re-run
    for c, v in zip(chunks, vectors):
        db.add(Chunk(document_id=doc.id, workspace_id=doc.workspace_id,
                     seq=c["seq"], text=c["text"], page=c["page"],
                     section=c["section"], embedding=json.dumps(v),
                     embedding_model=model))
    db.flush()
    if vectorstore.usable(model, len(vectors[0])):
        ids = [c.id for c in db.query(Chunk).filter_by(document_id=doc.id).order_by(Chunk.seq)]
        vectorstore.store_vectors(db, ids, vectors)
    doc.page_count = max((p for p, _ in pages), default=0)
    doc.status = "done"
    doc.error = ""
    return len(chunks)


# --- claiming / finishing ----------------------------------------------------

def claim_job(db, job_id: Optional[int] = None,
              worker_id: str = WORKER_ID) -> Optional[IngestionJob]:
    """Atomically claim one runnable job (or the given one). None if none."""
    now = _now()
    q = db.query(IngestionJob.id).filter(IngestionJob.status == "pending")
    q = q.filter((IngestionJob.next_run_at.is_(None)) | (IngestionJob.next_run_at <= now))
    if job_id is not None:
        q = q.filter(IngestionJob.id == job_id)
    for (jid,) in q.order_by(IngestionJob.id).limit(10).all():
        res = db.execute(
            update(IngestionJob)
            .where(IngestionJob.id == jid, IngestionJob.status == "pending")
            .values(status="processing", locked_at=now, worker_id=worker_id,
                    attempts=IngestionJob.attempts + 1))
        db.commit()
        if res.rowcount == 1:
            job = db.query(IngestionJob).filter_by(id=jid).first()
            db.refresh(job)
            return job
    return None


def _finish(db, job: IngestionJob, **values) -> bool:
    """Update the job only if this worker still holds the claim."""
    res = db.execute(
        update(IngestionJob)
        .where(IngestionJob.id == job.id, IngestionJob.status == "processing",
               IngestionJob.worker_id == job.worker_id)
        .values(**values))
    return res.rowcount == 1


def run_job(job_id: Optional[int] = None, worker_id: str = WORKER_ID) -> Optional[str]:
    """Claim and run one job. Returns its final status, or None if nothing ran."""
    db = SessionLocal()
    try:
        job = claim_job(db, job_id, worker_id)
        if job is None:
            return None
        jid, doc_id, attempt = job.id, job.document_id, job.attempts
        jlog("job_started", job=jid, document=doc_id, attempt=attempt, worker=worker_id)
        db.query(Document).filter_by(id=doc_id).update({"status": "processing"})
        db.commit()
        t0 = _now()
        try:
            n = process_document(db, doc_id)
            if not _finish(db, job, status="done", error="", locked_at=None):
                db.rollback()  # lost the claim (recovered by someone else)
                jlog("job_claim_lost", job=jid)
                return None
            db.commit()
            jlog("job_done", job=jid, document=doc_id, chunks=n,
                 seconds=round((_now() - t0).total_seconds(), 3))
            return "done"
        except Exception as e:  # noqa: BLE001 - every failure is recorded
            db.rollback()
            permanent = isinstance(e, PermanentError)
            msg = str(e)[:500]
            give_up = permanent or attempt >= job.max_attempts
            if give_up:
                ok = _finish(db, job, status="failed", error=msg, locked_at=None)
                db.query(Document).filter_by(id=doc_id).update(
                    {"status": "failed", "error": msg})
                status = "failed"
            else:
                delay = settings.job_backoff_base_s * (2 ** (attempt - 1))
                ok = _finish(db, job, status="pending", error=msg, locked_at=None,
                             next_run_at=_now() + timedelta(seconds=delay))
                db.query(Document).filter_by(id=doc_id).update({"status": "pending"})
                status = "retry"
            db.commit()
            jlog("job_failed" if give_up else "job_retry_scheduled", job=jid,
                 document=doc_id, attempt=attempt, permanent=permanent,
                 error=msg, claim_held=ok,
                 **({} if give_up else {"retry_in_s": delay}))
            return status
    finally:
        db.close()


def recover_stale(db, stale_after_s: Optional[float] = None) -> int:
    """Requeue (or fail) jobs whose worker vanished. Returns jobs touched."""
    limit = _now() - timedelta(seconds=stale_after_s if stale_after_s is not None
                               else settings.job_stale_after_s)
    stuck = (db.query(IngestionJob)
             .filter(IngestionJob.status == "processing", IngestionJob.locked_at < limit)
             .all())
    n = 0
    for job in stuck:
        give_up = job.attempts >= job.max_attempts
        res = db.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job.id, IngestionJob.status == "processing",
                   IngestionJob.locked_at == job.locked_at)  # still the same claim
            .values(status="failed" if give_up else "pending", locked_at=None,
                    worker_id="", next_run_at=None,
                    error="worker died mid-job" if give_up else job.error))
        if res.rowcount == 1:
            n += 1
            db.query(Document).filter_by(id=job.document_id).update(
                {"status": "failed" if give_up else "pending",
                 **({"error": "worker died mid-job"} if give_up else {})})
            jlog("job_recovered", job=job.id, failed=give_up, attempts=job.attempts)
    db.commit()
    return n


def pending_job_ids(db) -> list:
    now = _now()
    rows = (db.query(IngestionJob.id).filter(IngestionJob.status == "pending")
            .filter((IngestionJob.next_run_at.is_(None)) | (IngestionJob.next_run_at <= now))
            .order_by(IngestionJob.id).all())
    return [r[0] for r in rows]


def enqueue(job_id: int) -> None:
    """Hand a job to the configured queue. The db poller needs no push."""
    if settings.queue_backend == "celery":
        from .tasks import ingest_job
        try:
            ingest_job.delay(job_id)
        except Exception:  # noqa: BLE001 - broker down: the sweep will find it
            log.exception("could not enqueue job %s; it stays pending", job_id)


# --- re-embedding after a model change ------------------------------------

def reembed_stale(db, batch_size: int = 64) -> int:
    """Re-embed chunks whose embedding_model differs from the active backend."""
    current = backend_name()
    stale = (db.query(Chunk).filter(Chunk.embedding_model != current)
             .order_by(Chunk.id).all())
    done = 0
    for i in range(0, len(stale), batch_size):
        batch = stale[i:i + batch_size]
        vectors = embed_many([c.text for c in batch])
        for c, v in zip(batch, vectors):
            c.embedding = json.dumps(v)
            c.embedding_model = current
        db.flush()
        if vectorstore.usable(current, len(vectors[0])):
            vectorstore.store_vectors(db, [c.id for c in batch], vectors)
        db.commit()
        done += len(batch)
    if done:
        jlog("reembedded", chunks=done, model=current)
    return done


# --- threaded poller (QUEUE_BACKEND=db) ------------------------------------

def _loop(interval: float) -> None:
    migrated = False
    while not _stop.is_set():
        try:
            db = SessionLocal()
            try:
                if not migrated:
                    migrated = True
                    try:
                        reembed_stale(db)
                    except Exception:  # noqa: BLE001
                        log.exception("stale-chunk re-embedding failed")
                recover_stale(db)
            finally:
                db.close()
            while run_job() is not None and not _stop.is_set():
                pass  # drain everything runnable, then sleep
        except Exception:  # noqa: BLE001 - the worker must never die silently
            log.exception("worker tick failed")
        _stop.wait(interval)


def start_worker(interval: float = 1.0) -> threading.Thread:
    _stop.clear()
    thread = threading.Thread(target=_loop, args=(interval,), daemon=True,
                              name="nexus-ingestion-worker")
    thread.start()
    return thread


def stop_worker() -> None:
    _stop.set()

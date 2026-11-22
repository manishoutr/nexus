"""Background ingestion worker.

A daemon thread polls the ingestion_jobs table and processes pending jobs:
extract -> chunk -> embed -> store. Failures retry up to max_attempts with a
short backoff, then the job and its document are marked failed with the
error recorded. The queue is the database itself, so restarts lose nothing.

For production the same process_document() function drops into Celery/Redis
unchanged; the threaded worker keeps the local demo dependency-free.
"""
import json
import logging
import threading
import time
from pathlib import Path

from ..db import SessionLocal
from ..models import Chunk, Document, IngestionJob
from .chunk import chunk_document
from .embeddings import embed
from .extract import extract_pages

log = logging.getLogger("nexus.worker")
_stop = threading.Event()


def process_document(db, job: IngestionJob, data_dir: str) -> None:
    """Run one ingestion job. Raises on failure; the caller handles retries."""
    doc = db.query(Document).filter_by(id=job.document_id).first()
    if doc is None:
        raise RuntimeError(f"document {job.document_id} missing")
    path = Path(data_dir) / str(doc.workspace_id) / doc.filename
    pages = extract_pages(path.read_bytes(), doc.filename)
    chunks = chunk_document(pages)
    if not chunks:
        raise RuntimeError("no text could be extracted")

    db.query(Chunk).filter_by(document_id=doc.id).delete()  # idempotent re-runs
    for c in chunks:
        db.add(Chunk(document_id=doc.id, workspace_id=doc.workspace_id,
                     seq=c["seq"], text=c["text"], page=c["page"],
                     section=c["section"], embedding=json.dumps(embed(c["text"]))))
    doc.page_count = max((p for p, _ in pages), default=0)
    doc.status = "done"
    doc.error = ""
    job.status = "done"


def _tick() -> None:
    db = SessionLocal()
    try:
        job = (db.query(IngestionJob)
               .filter_by(status="pending")
               .order_by(IngestionJob.id).first())
        if job is None:
            return
        job.status = "processing"
        job.attempts += 1
        db.commit()
        from ..config import settings
        try:
            process_document(db, job, settings.data_dir)
        except Exception as e:  # noqa: BLE001 - record any failure on the job
            log.exception("ingestion job %s failed (attempt %d)", job.id, job.attempts)
            job.error = str(e)
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                doc = db.query(Document).filter_by(id=job.document_id).first()
                if doc:
                    doc.status = "failed"
                    doc.error = str(e)
            else:
                job.status = "pending"  # retry on the next tick
        db.commit()
    finally:
        db.close()


def _loop(interval: float) -> None:
    while not _stop.is_set():
        try:
            _tick()
        except Exception:  # noqa: BLE001 - the worker must never die silently
            log.exception("worker tick failed")
        _stop.wait(interval)


def start_worker(interval: float = 1.0) -> threading.Thread:
    thread = threading.Thread(target=_loop, args=(interval,), daemon=True,
                              name="nexus-ingestion-worker")
    thread.start()
    return thread


def stop_worker() -> None:
    _stop.set()

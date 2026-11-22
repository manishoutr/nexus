"""Phase 3: storage, upload validation, atomic claims, retries, recovery."""
import hashlib
import io
import threading
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.db import Base, SessionLocal, engine, init_db
from app.main import app
from app.models import Chunk, Document, IngestionJob, Workspace
from app.services import worker
from app.services.storage import LocalStorage, get_storage, new_key
from app.services.validate import UploadRejected, validate_upload

TXT = b"Hybrid retrieval combines BM25 keyword ranking with vector similarity. " * 5


@pytest.fixture(autouse=True)
def fresh():
    worker.stop_worker()
    Base.metadata.drop_all(engine)
    init_db()
    db = SessionLocal()
    db.add(Workspace(name="t"))
    db.commit()
    db.close()
    yield


@pytest.fixture
def client():
    # No `with`: skips startup so the background poller is NOT running and the
    # tests drive the worker explicitly.
    return TestClient(app)


def upload(client, name, data, ws=1):
    return client.post(f"/api/workspaces/{ws}/documents", files={"file": (name, data)})


# --- validation --------------------------------------------------------------

def test_rejects_wrong_signature_empty_and_binary_text():
    with pytest.raises(UploadRejected) as e:
        validate_upload("a.pdf", b"not a pdf at all", 10**6)
    assert e.value.status == 415
    with pytest.raises(UploadRejected) as e:
        validate_upload("a.txt", b"", 10**6)
    assert e.value.status == 422
    with pytest.raises(UploadRejected):
        validate_upload("a.txt", b"abc\x00def", 10**6)
    with pytest.raises(UploadRejected):
        validate_upload("a.docx", b"PK\x03\x04garbage", 10**6)
    with pytest.raises(UploadRejected) as e:
        validate_upload("a.txt", b"x" * 100, 50)
    assert e.value.status == 413
    with pytest.raises(UploadRejected):
        validate_upload("a.exe", b"MZ", 10**6)


def test_encrypted_and_corrupt_pdf_rejected():
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(UploadRejected) as e:
        validate_upload("enc.pdf", buf.getvalue(), 10**6)
    assert "password" in e.value.message
    with pytest.raises(UploadRejected) as e:
        validate_upload("bad.pdf", b"%PDF-1.4\nbroken", 10**6)
    assert "corrupt" in e.value.message


def test_api_returns_validation_errors(client):
    assert upload(client, "x.pdf", b"hello").status_code == 415
    assert upload(client, "x.txt", b"").status_code == 422
    assert upload(client, "x.txt", b"a" * (11 * 1024 * 1024)).status_code == 413
    assert upload(client, "x.txt", TXT, ws=99).status_code == 404


# --- storage / uuid keys / dedupe -------------------------------------------

def test_duplicate_filenames_different_content_do_not_collide(client):
    a = upload(client, "notes.txt", TXT + b" one")
    b = upload(client, "notes.txt", TXT + b" two")
    assert a.status_code == b.status_code == 201
    db = SessionLocal()
    docs = db.query(Document).all()
    assert len({d.storage_key for d in docs}) == 2
    assert all("notes" not in d.storage_key for d in docs)  # key is uuid, not name
    assert {d.filename for d in docs} == {"notes.txt"}
    assert get_storage().get(docs[0].storage_key) != get_storage().get(docs[1].storage_key)


def test_same_bytes_twice_is_idempotent(client):
    a = upload(client, "a.txt", TXT)
    b = upload(client, "renamed.txt", TXT)
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    assert SessionLocal().query(IngestionJob).count() == 1


def test_concurrent_identical_uploads_create_one_document(client):
    codes = []

    def go():
        codes.append(upload(client, "c.txt", TXT + b" concurrent").status_code)

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(set(codes)) in ([200, 201], [201])
    assert codes.count(201) == 1
    assert SessionLocal().query(Document).count() == 1
    assert SessionLocal().query(IngestionJob).count() == 1


def test_path_traversal_filename_is_harmless(client):
    r = upload(client, "../../etc/evil.txt", TXT)
    assert r.status_code == 201
    assert r.json()["filename"] == "evil.txt"
    with pytest.raises(ValueError):
        LocalStorage("/tmp/x")._path("../../etc/passwd")


# --- end to end + delete ------------------------------------------------------

def test_ingest_then_delete_removes_everything(client):
    doc_id = upload(client, "n.txt", TXT).json()["id"]
    assert worker.run_job() == "done"
    db = SessionLocal()
    doc = db.query(Document).get(doc_id)
    assert doc.status == "done" and db.query(Chunk).filter_by(document_id=doc_id).count() > 0
    key = doc.storage_key
    assert get_storage().exists(key)
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    db = SessionLocal()
    assert db.query(Document).count() == 0
    assert db.query(Chunk).count() == 0
    assert db.query(IngestionJob).count() == 0
    assert not get_storage().exists(key)
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


# --- claims ------------------------------------------------------------------

def _add_doc_job(db, name="d.txt", data=TXT, max_attempts=3):
    key = new_key(1)
    get_storage().put(key, data)
    d = Document(workspace_id=1, filename=name, storage_key=key,
                 content_hash=hashlib.sha256(data + name.encode()).hexdigest(),
                 size_bytes=len(data), status="pending")
    db.add(d)
    db.flush()
    j = IngestionJob(document_id=d.id, max_attempts=max_attempts)
    db.add(j)
    db.commit()
    return d.id, j.id


def test_concurrent_workers_claim_each_job_once():
    db = SessionLocal()
    ids = [_add_doc_job(db, f"d{i}.txt")[1] for i in range(12)]
    db.close()
    claimed = []
    lock = threading.Lock()

    def w(n):
        s = SessionLocal()
        while True:
            j = worker.claim_job(s, worker_id=f"w{n}")
            if j is None:
                break
            with lock:
                claimed.append(j.id)
        s.close()

    ts = [threading.Thread(target=w, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(claimed) == sorted(ids)  # every job exactly once, none twice


def test_two_workers_running_the_same_job_only_one_runs():
    db = SessionLocal()
    _, jid = _add_doc_job(db)
    db.close()
    results = []
    ts = [threading.Thread(target=lambda: results.append(worker.run_job(jid)))
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("done") == 1 and results.count(None) == 3
    assert SessionLocal().query(Chunk).count() > 0
    ch = SessionLocal().query(Chunk).count()
    assert ch == len({c.seq for c in SessionLocal().query(Chunk)})  # no duplicates


# --- retries / backoff / permanent failures ---------------------------------

def test_transient_failure_retries_with_backoff_then_succeeds(monkeypatch):
    db = SessionLocal()
    did, jid = _add_doc_job(db)
    calls = {"n": 0}
    real = worker.embed_many

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("embedding service hiccup")
        return real(texts)

    monkeypatch.setattr(worker, "embed_many", flaky)
    assert worker.run_job() == "retry"
    j = SessionLocal().query(IngestionJob).get(jid)
    assert j.status == "pending" and j.attempts == 1 and j.next_run_at > datetime.utcnow()
    assert worker.run_job() is None  # backoff: not claimable yet
    time.sleep(0.25)
    assert worker.run_job() == "retry"  # second failure, delay doubles
    j2 = SessionLocal().query(IngestionJob).get(jid)
    gap1 = 0.2
    assert (j2.next_run_at - datetime.utcnow()).total_seconds() > gap1  # ~0.4s
    time.sleep(0.5)
    assert worker.run_job() == "done"
    j3 = SessionLocal().query(IngestionJob).get(jid)
    assert j3.status == "done" and j3.attempts == 3 and j3.error == ""


def test_retries_exhausted_marks_failed(monkeypatch):
    db = SessionLocal()
    did, jid = _add_doc_job(db, max_attempts=2)
    monkeypatch.setattr(worker, "embed_many", lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    assert worker.run_job() == "retry"
    time.sleep(0.25)
    assert worker.run_job() == "failed"
    s = SessionLocal()
    assert s.query(IngestionJob).get(jid).status == "failed"
    d = s.query(Document).get(did)
    assert d.status == "failed" and "down" in d.error


def test_permanent_error_fails_immediately_without_retry():
    db = SessionLocal()
    did, jid = _add_doc_job(db, "bad.pdf", b"%PDF-1.4 junk")
    assert worker.run_job() == "failed"
    s = SessionLocal()
    j = s.query(IngestionJob).get(jid)
    assert j.attempts == 1 and j.status == "failed"
    assert "corrupt" in s.query(Document).get(did).error


def test_missing_stored_file_is_permanent():
    db = SessionLocal()
    did, jid = _add_doc_job(db)
    get_storage().delete(db.query(Document).get(did).storage_key)
    assert worker.run_job() == "failed"
    assert "missing" in SessionLocal().query(Document).get(did).error


# --- crash recovery ----------------------------------------------------------

def test_stale_processing_job_is_recovered_and_completes():
    db = SessionLocal()
    did, jid = _add_doc_job(db)
    # simulate a worker that claimed the job and died
    claimed = worker.claim_job(db, jid, worker_id="dead-worker")
    assert claimed.status == "processing"
    assert worker.recover_stale(db, stale_after_s=3600) == 0   # not stale yet
    db.query(IngestionJob).filter_by(id=jid).update(
        {"locked_at": datetime.utcnow() - timedelta(hours=1)})
    db.commit()
    assert worker.recover_stale(db, stale_after_s=300) == 1
    assert db.query(IngestionJob).get(jid).status == "pending"
    assert worker.run_job() == "done"
    assert SessionLocal().query(IngestionJob).get(jid).attempts == 2


def test_stale_job_out_of_attempts_is_failed_not_looped():
    db = SessionLocal()
    did, jid = _add_doc_job(db, max_attempts=1)
    worker.claim_job(db, jid, worker_id="dead")
    db.query(IngestionJob).filter_by(id=jid).update(
        {"locked_at": datetime.utcnow() - timedelta(hours=1)})
    db.commit()
    worker.recover_stale(db, stale_after_s=300)
    assert db.query(IngestionJob).get(jid).status == "failed"
    assert db.query(Document).get(did).status == "failed"


def test_crash_after_chunks_written_leaves_no_duplicates(monkeypatch):
    """A retry after a partial run replaces chunks rather than appending."""
    db = SessionLocal()
    did, jid = _add_doc_job(db)
    assert worker.run_job() == "done"
    first = SessionLocal().query(Chunk).filter_by(document_id=did).count()
    s = SessionLocal()
    s.query(IngestionJob).filter_by(id=jid).update({"status": "pending", "worker_id": ""})
    s.commit()
    assert worker.run_job() == "done"
    assert SessionLocal().query(Chunk).filter_by(document_id=did).count() == first


def test_stale_writer_cannot_overwrite_new_claim():
    """Worker A is declared dead and B takes over; A's late result is dropped."""
    db = SessionLocal()
    did, jid = _add_doc_job(db)
    from types import SimpleNamespace
    a = worker.claim_job(db, jid, worker_id="A")
    a = SimpleNamespace(id=a.id, worker_id="A")  # snapshot: the ORM row is shared
    db.query(IngestionJob).filter_by(id=jid).update(
        {"locked_at": datetime.utcnow() - timedelta(hours=1)})
    db.commit()
    worker.recover_stale(db, 300)
    b = worker.claim_job(db, jid, worker_id="B")
    assert b is not None
    assert worker._finish(db, a, status="done") is False   # A no longer owns it
    db.commit()
    assert db.query(IngestionJob).get(jid).status == "processing"

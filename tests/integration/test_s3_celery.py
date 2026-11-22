"""Integration tests against real servers (run locally, no mocks of the
protocols involved):

* S3: moto's standalone S3 server (speaks the real S3 HTTP API). This is an
  S3-compatible stand-in, NOT AWS S3.
* Queue: a real Redis server (redislite bundles Redis 6.2) and a real Celery
  worker process.

Skipped automatically when redislite / moto / celery are not installed.
"""
import hashlib
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("celery")
redislite = pytest.importorskip("redislite")
moto_server = pytest.importorskip("moto.server")

from app.auth import make_token
from app.db import SessionLocal, init_db, reset_db_for_tests  # noqa: E402
from app.models import Chunk, Document, IngestionJob, User, Workspace  # noqa: E402
from app.services import storage as storage_mod  # noqa: E402
from app.services import worker  # noqa: E402
from app.services.storage import S3Storage, new_key  # noqa: E402

TXT = b"Celery workers pull ingestion jobs from Redis. Each job is claimed in the database first. " * 4


@pytest.fixture(autouse=True)
def fresh():
    reset_db_for_tests()
    init_db()
    db = SessionLocal()
    db.add(User(email="owner@example.com", password_hash="x"))
    db.flush()
    db.add(Workspace(name="t", owner_id=1))
    db.commit()
    db.close()
    yield


# --- S3 ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def s3_endpoint():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    srv = moto_server.ThreadedMotoServer(port=5055, verbose=False)
    srv.start()
    yield "http://127.0.0.1:5055"
    srv.stop()


def test_s3_storage_roundtrip_and_missing_key(s3_endpoint):
    s3 = S3Storage("nexus-test", s3_endpoint)
    s3.ensure_bucket()
    k = new_key(1)
    s3.put(k, b"hello")
    assert s3.exists(k) and s3.get(k) == b"hello"
    s3.delete(k)
    s3.delete(k)  # idempotent
    assert not s3.exists(k)
    with pytest.raises(FileNotFoundError):
        s3.get(k)


def test_ingest_from_s3_then_delete(s3_endpoint, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    s3 = S3Storage("nexus-test2", s3_endpoint)
    s3.ensure_bucket()
    monkeypatch.setattr(storage_mod, "_storage", s3)
    client = TestClient(app, headers={"Authorization": "Bearer " + make_token(1, "access")})
    r = client.post("/api/workspaces/1/documents", files={"file": ("a.txt", TXT)})
    assert r.status_code == 201
    doc_id = r.json()["id"]
    key = SessionLocal().query(Document).get(doc_id).storage_key
    assert s3.exists(key)                       # bytes really live in the bucket
    assert worker.run_job() == "done"
    assert SessionLocal().query(Chunk).count() > 0
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert not s3.exists(key)


# --- Redis + Celery ------------------------------------------------------------

@pytest.fixture(scope="module")
def redis_server(tmp_path_factory):
    r = redislite.Redis(str(tmp_path_factory.mktemp("redis") / "r.db"),
                        serverconfig={"port": "6391", "bind": "127.0.0.1"})
    assert r.ping()
    yield r
    r.shutdown()


@pytest.fixture
def celery_worker(redis_server):
    from tests.conftest import TEST_ENV
    env = dict(os.environ, **TEST_ENV, QUEUE_BACKEND="celery")
    p = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "app.services.tasks", "worker",
         "-l", "info", "--concurrency=2", "-P", "threads"],
        env=env, stdout=open("/tmp/celery-test-worker.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    ok = False
    while time.time() - t0 < 30:
        from app.services.tasks import celery_app
        if celery_app.control.ping(timeout=1):
            ok = True
            break
    assert ok, "celery worker did not start"
    yield p
    p.terminate()
    p.wait(timeout=15)


def _wait(pred, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.2)
    return False


def test_upload_is_processed_by_real_celery_worker(celery_worker, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app
    monkeypatch.setattr(settings, "queue_backend", "celery")
    client = TestClient(app, headers={"Authorization": "Bearer " + make_token(1, "access")})
    ids = []
    for i in range(5):
        r = client.post("/api/workspaces/1/documents",
                        files={"file": (f"f{i}.txt", TXT + str(i).encode())})
        assert r.status_code == 201
        ids.append(r.json()["id"])
    assert _wait(lambda: all(
        SessionLocal().query(Document).get(i).status == "done" for i in ids))
    db = SessionLocal()
    jobs = db.query(IngestionJob).all()
    assert len(jobs) == 5 and all(j.status == "done" and j.attempts == 1 for j in jobs)
    assert not db.query(IngestionJob).filter(IngestionJob.worker_id == "").count()
    assert db.query(Chunk).count() >= 5


def test_duplicate_task_delivery_does_not_double_process(celery_worker):
    from app.services.tasks import ingest_job
    db = SessionLocal()
    key = new_key(1)
    storage_mod.get_storage().put(key, TXT)
    d = Document(workspace_id=1, filename="dup.txt", storage_key=key,
                 content_hash=hashlib.sha256(TXT).hexdigest(), status="pending")
    db.add(d)
    db.flush()
    j = IngestionJob(document_id=d.id)
    db.add(j)
    db.commit()
    for _ in range(6):                      # broker delivers the same job 6 times
        ingest_job.delay(j.id)
    assert _wait(lambda: SessionLocal().query(IngestionJob).get(j.id).status == "done")
    time.sleep(1)
    s = SessionLocal()
    assert s.query(IngestionJob).get(j.id).attempts == 1
    seqs = [c.seq for c in s.query(Chunk).filter_by(document_id=d.id)]
    assert len(seqs) == len(set(seqs))


def test_sweep_recovers_job_orphaned_by_dead_worker(celery_worker):
    from datetime import datetime, timedelta
    from app.services.tasks import sweep
    db = SessionLocal()
    key = new_key(1)
    storage_mod.get_storage().put(key, TXT + b" orphan")
    d = Document(workspace_id=1, filename="o.txt", storage_key=key,
                 content_hash="orphan", status="processing")
    db.add(d)
    db.flush()
    j = IngestionJob(document_id=d.id, status="processing", attempts=1,
                     worker_id="dead", locked_at=datetime.utcnow() - timedelta(hours=2))
    db.add(j)
    db.commit()
    sweep.delay()
    assert _wait(lambda: SessionLocal().query(IngestionJob).get(j.id).status == "done")
    assert SessionLocal().query(IngestionJob).get(j.id).attempts == 2
    assert SessionLocal().query(Document).get(d.id).status == "done"


def test_rate_limit_counts_are_shared_across_processes_via_redis(redis_server):
    """Two limiter instances (as in two API workers) on one Redis share counts.
    Uses the same `limits` storage/strategy slowapi builds from RATE_LIMIT_STORAGE."""
    from slowapi import Limiter
    mk = lambda: Limiter(key_func=lambda r: "k", storage_uri="redis://127.0.0.1:6391/1")
    a, b = mk(), mk()
    from limits import parse
    item = parse("3/minute")
    a.limiter.storage.reset()
    results = [a.limiter.hit(item, "user:1"), b.limiter.hit(item, "user:1"),
               a.limiter.hit(item, "user:1"), b.limiter.hit(item, "user:1")]
    assert results == [True, True, True, False]   # 4th hit blocked, from the other worker

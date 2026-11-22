"""Celery app and tasks (QUEUE_BACKEND=celery, broker = Redis).

Run a worker:   celery -A app.services.tasks worker -l info
Run the sweep:  celery -A app.services.tasks beat -l info

The task body is just run_job(): the claim in the database decides who does
the work, so a duplicate delivery or a second worker is harmless.
"""
from celery import Celery

from ..config import settings
from ..db import SessionLocal, init_db

celery_app = Celery("nexus", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_acks_late=True,              # unacked tasks return to the queue if a worker dies
    worker_prefetch_multiplier=1,
    task_track_started=True,
    beat_schedule={
        "sweep-jobs": {"task": "nexus.sweep", "schedule": 30.0},
    },
)


@celery_app.task(name="nexus.ingest_job")
def ingest_job(job_id: int):
    from .worker import run_job
    init_db()
    return run_job(job_id)


@celery_app.task(name="nexus.sweep")
def sweep():
    """Safety net: requeue orphaned jobs and push anything pending that was
    never enqueued (broker was down at upload time) or whose backoff elapsed."""
    from .worker import pending_job_ids, recover_stale
    init_db()
    db = SessionLocal()
    try:
        recovered = recover_stale(db)
        ids = pending_job_ids(db)
    finally:
        db.close()
    for jid in ids:
        ingest_job.delay(jid)
    return {"recovered": recovered, "enqueued": len(ids)}

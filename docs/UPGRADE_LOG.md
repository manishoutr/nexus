# Engineering upgrade log

One section per phase: what changed, files, decisions, tests, and what is still not proven.

## Phase 1: audit
No code changed. Found a clean ~880 line layout. Gaps against the target design: hashing-only embeddings, scores mixed into one number, DB-polling worker without atomic claims, local-disk storage, no auth, SQLite only, no migrations, no CI.

## Phase 2: retrieval
**Changed**
- Real embeddings: sentence-transformers `all-MiniLM-L6-v2` (384-dim, normalized, lazy load). The 256-dim hashing embedder remains as a fallback so the light build still runs. `EMBEDDING_BACKEND=auto|st|hash`.
- BM25 and vector rankings are computed independently and merged with reciprocal rank fusion (k=60). Top-k is configurable (`RETRIEVAL_K`, default 5).
- Abstention uses a separate absolute `strength` signal, because RRF scores only encode rank and cannot say "nothing relevant". Threshold is per backend (ST 0.30).
- Each chunk stores `embedding_model`; on startup the worker re-embeds chunks built with a different model.
- Optional cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `RERANK_ENABLED`), fails open to RRF order.

**Files created:** `app/services/rerank.py`, `tests/test_retrieval_st.py`, `requirements-ml.txt`
**Files modified:** `embeddings.py`, `retrieval.py`, `config.py`, `models.py`, `db.py`, `worker.py`, `qa.py`, `main.py`, `tests/test_retrieval.py`, `tests/test_api.py`, `tests/test_qa.py`
**Removed:** none

**Decisions:** ST dependencies live in `requirements-ml.txt` so the default Docker/Render image stays small. A small ALTER shim in `db.py` adds the new column until Alembic (Phase 6).

**Tests:** 26 passed. Includes real-model tests (384 dims, unit norm, paraphrase match, abstention on an unrelated query) and a manual model-migration check (12 chunks re-embedded).

**Not proven:** cross-encoder rerank has not been run against the real model. Vectors are still stored as JSON, pgvector comes in Phase 6. The live Render site still runs the hashing build; torch is unlikely to fit the 512 MB free tier (unverified).

## Phase 3: file processing and background reliability
**Changed**
- Storage interface (`app/services/storage.py`) with `LocalStorage` and `S3Storage` (boto3, any S3-compatible endpoint). Files are stored under `<workspace>/<uuid>`; the original filename is metadata only. Local writes are atomic (temp file + rename) and reject `../` keys.
- Upload validation by content (`validate.py`): extension allowlist, PDF/DOCX signatures, zip structure for DOCX, encrypted or corrupt or page-less PDFs, binary or non-UTF-8 text, empty and whitespace-only files, size limit. Errors map to 413/415/422.
- Idempotent uploads: sha256 per workspace, unique index `(workspace_id, content_hash)`. Same bytes again returns the existing document (200). Concurrent identical uploads resolve through the unique index (one 201, rest 200).
- `DELETE /api/documents/{id}` removes chunks, jobs, the document row and the stored object (rows first, object second).
- Job queue (`worker.py`): atomic claim by conditional UPDATE (`pending -> processing` succeeds for one caller), exponential backoff (`next_run_at`, base `JOB_BACKOFF_BASE_S`, doubling), permanent errors (corrupt file, missing file, deleted document) fail at once, stale `processing` jobs are requeued or failed after `JOB_STALE_AFTER_S`, a worker that lost its claim cannot overwrite the result, chunk replacement is one transaction so retries leave no duplicates. One JSON log line per event (`job_started`, `job_done`, `job_retry_scheduled`, `job_failed`, `job_recovered`).
- Celery + Redis (`tasks.py`, `QUEUE_BACKEND=celery`): `ingest_job` task runs the same `run_job`, `acks_late`, prefetch 1, plus a 30 s `sweep` beat task that recovers orphaned jobs and pushes jobs that were never enqueued (broker down at upload). Default `QUEUE_BACKEND=db` keeps the threaded poller so the demo needs no extra services.

**Files created:** `app/services/storage.py`, `validate.py`, `tasks.py`; `tests/conftest.py`, `tests/test_reliability.py`, `tests/integration/test_s3_celery.py`, `tests/__init__.py`
**Files modified:** `app/models.py`, `db.py`, `config.py`, `main.py`, `services/worker.py`, `requirements.txt` (boto3, celery[redis]), `requirements-dev.txt`
**Removed:** none (the old per-filename disk layout is replaced; legacy rows without a storage key still resolve to the old path)

**Tests:** 49 passed (`pytest tests`): 26 earlier + 18 reliability + 5 integration.
- Reliability (SQLite): duplicate filenames, same bytes twice, 8 concurrent identical uploads, path traversal, 6 threads claiming 12 jobs (each exactly once), 4 workers on one job (one runs), retry with doubling backoff then success, retries exhausted, permanent failures, stale recovery, stale writer rejected, rerun leaves no duplicate chunks, delete cleans storage.
- Integration, real servers: S3 API served by moto's standalone server, a real Redis 6.2 (redislite) and a real Celery worker process. 5 uploads processed through Redis; 6 duplicate deliveries of one task ran once; an orphaned job was recovered by the sweep task.

**Not proven**
- S3 evidence is an S3-compatible server (moto), not AWS S3. No AWS account was used.
- Concurrency tests ran on SQLite. The same claim SQL runs on Postgres but is only verified there in Phase 6.
- A real `kill -9` of a Celery worker mid-job was not tested; stale recovery is tested by simulating the dead claim.
- Redis visibility timeout and Celery redelivery behaviour were not exercised.
- Running the integration tests needs ports 6391 and 5055 free. A leftover redis-server from a killed run makes them error until it is stopped.

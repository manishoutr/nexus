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

## Phase 4: auth and security
**Changed**
- Accounts: `POST /api/auth/register|login|refresh`, `GET /api/auth/me`. Passwords use scrypt with a random salt (stdlib). JWTs are HS256 with the algorithm pinned on decode, a `type` claim (access 30 min, refresh 7 days; a refresh token is rejected as an access token and the reverse), `exp` and a `jti`. `JWT_SECRET` comes from the environment; without it a random per-process secret is used and a warning is logged.
- Login errors are identical for unknown email and wrong password.
- Workspace authorization: `owner_id`, a `memberships` table and roles viewer < editor < owner (`POST /api/workspaces/{id}/members`, owner only). Every workspace and document route checks the role. A user with no access to a private workspace gets 404 (same body as a nonexistent id), so ids can't be probed. A viewer who tries to write gets 403.
- The seeded demo workspace is public: anyone can search and ask, only logged-in members can write.
- Rate limiting with slowapi: auth routes 10/min, search/ask 20/min, uploads 30/min (env-configurable). Keyed by user id from the token, otherwise by IP. Storage is in-memory per process by default; `RATE_LIMIT_STORAGE=redis://...` shares limits across processes (not yet tested against Redis).
- Prompt-injection safeguards (`services/safety.py`): passages are wrapped in `<passage>` tags and the system prompt says they are untrusted data; our delimiter tags inside documents are escaped; sentences matching known injection phrasings are replaced with a marker before reaching the model; the model's `used_chunk_ids` are range-checked and answer length is capped; the model has no tools. The offline answer never echoes injection-looking sentences. The response reports `flagged_passages`. Questions are capped at 2000 characters and stripped of control characters; `k` is bounded 1..20.
- `/api/metrics` now needs login.

**Files created:** `app/auth.py`, `app/ratelimit.py`, `app/services/safety.py`, `tests/test_security.py`
**Files modified:** `models.py`, `db.py`, `config.py`, `schemas.py`, `main.py`, `services/qa.py`, `tests/conftest.py`, `tests/test_api.py`, `tests/test_reliability.py`, `tests/integration/test_s3_celery.py`, `requirements.txt`
**Removed:** none

**Tests:** 62 passed. 13 new security tests: hashing, register/login/me, token attacks (expired, wrong secret, `alg: none`, refresh-as-access, garbage), login enumeration, cross-user access to every workspace and document route (user B and anonymous), viewer/editor/owner matrix, public workspace behaviour, login rate limit (10 allowed, then 429), per-user rate limit isolation, injection sanitising, a hostile document through a fake LLM (delimiters escaped, instruction removed, out-of-range chunk ids ignored), offline answer with a hostile document, input limits.

**A bug the tests caught:** my first edit to `DELETE /api/documents/{id}` silently did not apply (a docstring differed), so any logged-in user could delete any document. The cross-user test failed on it and it is fixed. Error messages for private workspaces also leaked the id and now don't.

**Not proven / limits**
- The injection filter is a phrase heuristic. Paraphrased attacks pass it; the tag escaping, the "data not instructions" system prompt and the lack of tools are the other layers. None of it was tested against a real Claude model (no key, no spend), so the real-model resistance is unverified.
- Sentence-level removal also drops legitimate text that sits in the same sentence as an injection phrase.
- Refresh tokens are not revocable (no server-side list) and there is no logout or password reset or email verification.
- Rate limits were tested with the in-memory store only.
- The current vanilla-JS UI has no login screen. Anonymous visitors can use the public demo workspace; uploading needs a token. A login UI comes with the React frontend.


## Phase 5: AI quality and evaluation
**Changed**
- Citation verification (`services/citations.py`): every citation carries `chunk_id`, `document_id`, `document_name`, `page`, `excerpt` and is checked against the database before it is shown: chunk exists, same workspace, belongs to the named document, page matches chunk metadata, name matches the document row, excerpt really is chunk text. Inline references like `[file, page N]` inside an answer are checked against the retrieved passages and removed if they don't match. If no citation survives, the answer becomes an abstention. Duplicate citations collapse.
- Claude path: the prompt asks for `insufficient_evidence` and `used_chunk_ids` and forbids bracketed references in the text; ids are range-checked; an answer that cites no retrieved passage is discarded in favour of the offline answer, with a warning.
- LLM failures (`services/llm.py`): classified `LLMError` (timeout, rate_limited, auth, api_error, invalid_output). Timeouts, rate limits and 5xx retry with backoff (3 tries); auth and invalid output don't. `ask` catches the error and falls back to the extractive offline answer with a `warnings` entry. The API never returns a 500 for an LLM failure.
- Abstention: unchanged evidence-strength gate, plus (when `RERANK_ENABLED=true`) a cross-encoder gate: best passage score below `ABSTAIN_RERANK_THRESHOLD` (-1.5) means not answerable. Claude can also abstain itself. Threshold picked on the dev split only.
- Evaluation framework (`eval/`): `dataset.jsonl` (44 labelled questions: 22 single-document, 5 paraphrase, 5 multi-document, 7 near-miss unanswerable, 5 unrelated unanswerable; split dev/test), `metrics.py`, `run.py`. Relevance is defined by gold evidence phrases, so metrics survive re-chunking. Metrics: Precision@K, Recall@K, MRR for BM25, vector and hybrid; abstention accuracy with false-abstain and false-answer rates; citation validity and precision; a lexical faithfulness heuristic; answer-covers-evidence. Reports are JSON (`eval/results/report-*.json`); dashboard at `/eval.html`.

**Results** (offline answers, 12 chunks, k=5; `python -m eval.run --backend hash|st [--rerank]`):

| config | MRR | P@5 | R@5 | abstention acc. | wrongly abstained | wrongly answered | abstention acc. (test split) |
|---|---|---|---|---|---|---|---|
| hashing embeddings | 0.917 | 0.269 | 0.984 | 0.841 | 0.000 | 0.583 | 0.850 |
| MiniLM | 0.917 | 0.275 | 1.000 | 0.818 | 0.000 | 0.667 | 0.800 |
| MiniLM + cross-encoder gate | 0.969 | 0.275 | 1.000 | 0.886 | 0.094 | 0.167 | 0.850 |

- Hybrid beats either method alone on MRR only with MiniLM + reranker (0.969 vs 0.917). With hashing or plain MiniLM, hybrid, BM25 and vector are within noise on this set.
- Abstention on unrelated questions is solved by the strength gate. Near-miss questions (topical but unanswered in the text) fooled it: 58 to 67% were wrongly answered. The cross-encoder gate cuts that to 17% overall, at the price of wrongly abstaining on 9% of answerable questions (all paraphrases). On the test split accuracy is 85% either way; the threshold sweep is flat in the 80 to 85% band, so the threshold choice is inside the noise of 20 test questions.
- Weak spots shown on the dashboard: multi-document questions (answer covers 60% of evidence) and paraphrases (50%), because the offline answer is extractive and picks sentences by word overlap.

**Files created:** `app/services/citations.py`, `eval/__init__.py`, `eval/dataset.jsonl`, `eval/metrics.py`, `eval/run.py`, `eval/results/report-{hash,st,st-rerank}.json`, `app/static/eval.html`, `app/static/eval/report-*.json`, `tests/test_ai_quality.py`, `tests/test_eval.py`, `tests/test_rerank.py`
**Files modified:** `services/llm.py`, `services/qa.py`, `services/rerank.py`, `services/safety.py`, `schemas.py`, `config.py`, `auth.py` (production JWT check), `tests/test_security.py`, `tests/integration/test_s3_celery.py`
**Removed:** none

**Also in this change (parent request after Phase 4):** `APP_ENV=production` now refuses to start unless `JWT_SECRET` is at least 32 characters; development keeps the random fallback. New tests: public demo workspace write attempts (anonymous 401, other user 403) on documents/members; two limiter instances sharing a real Redis share counts (tests the `limits` storage the limiter uses, not a two-process HTTP run).

**Tests:** 85 passed (`pytest tests`, includes integration). New: 11 AI-quality tests (each kind of bad citation, fabricated inline reference, Claude timeout with 3 attempts then offline fallback, rate limit then success, auth and invalid-output not retried, fenced JSON, Claude abstention, ghost citation dropped, no-evidence question), 5 eval tests (metric maths by hand, dataset integrity, eval run reproducible and deterministic), 4 reranker tests against the real cross-encoder (orders the answering passage first, fails open when the model is unavailable or scoring raises).

**Not proven / limits**
- No real Claude call was ever made; all Claude-path tests use a fake client. The LLM-failure handling is tested against exception shapes, not the live API.
- The dataset is small and written by the developer after reading the three seed documents, with only 12 chunks, so R@5 is close to trivially high and P@5 is capped by chunk overlap. Differences between configurations are mostly within noise.
- The faithfulness metric is lexical and the offline answer is extractive, so it is near 1.0 by construction. It becomes informative only with Claude.
- The reranker needs a second model download (about 90 MB) and torch.

## Phase 6: infrastructure (IN PROGRESS, partial)
**Done and tested**
- Alembic (`alembic.ini`, `migrations/`): `0001` baseline is idempotent so databases created before Alembic are adopted without data loss; `0002` adds pgvector (`chunks.embedding_vec vector(384)` + HNSW cosine index, Postgres only). `init_db()` now runs `upgrade head`; the old ALTER shim is gone. DB URL comes from `DATABASE_URL`.
- PostgreSQL + pgvector: similarity search runs inside Postgres when the MiniLM backend is active (`services/vectorstore.py`); BM25 stays in Python; the hashing fallback and SQLite use the JSON vectors.
- Real-server testing without Docker: `python scripts/test_on_postgres.py` starts PostgreSQL 16.2 with pgvector 0.6.2 (pgserver package) and runs the suite on it.
- Readiness `/api/ready` (database, migrations at head, storage, Redis when Celery is on; 503 on failure) next to the cheap `/api/health`. Structured JSON logs with request ids (`LOG_FORMAT=json|text`).
- A bug found by running on Postgres: a NUL character in a question crashed the QA log insert. Questions are now cleaned before storage.

**Tests:** SQLite 85 passed + 5 skipped (the pgvector tests skip off Postgres); on PostgreSQL the suite passes (79 earlier run, plus 8 migration/pgvector tests passed). Migrations: fresh schema equals the models (alembic autogenerate diff empty), upgrade repeatable, downgrade to base, pre-Alembic database adopted with data kept. pgvector: extension and HNSW index exist, vectors have 384 dims, the planner uses the HNSW index when seqscan is disabled, retrieval goes through pgvector and ranks the same as the Python cosine path, workspace isolation.

**Infrastructure files (added)**
- `Dockerfile` (non-root, healthcheck, `WITH_ML=1` build arg for sentence-transformers), `docker-compose.yml` (api, celery worker, beat, pgvector Postgres, Redis, MinIO + bucket init), `.env.example`, `docs/ENVIRONMENT.md`.
- `.github/workflows/ci.yml`: ruff, tests on SQLite, tests on a pgvector Postgres service. `ruff.toml`, `pytest.ini` (tests/ only).
- **Not verified:** Docker is not installed here, so the Dockerfile and compose file have NOT been built or run (compose was only parsed as YAML). CI has not run on GitHub yet.

**Bug found by the real Celery integration test after the Alembic change:** the worker's thread pool called `init_db()` from several threads at once and Alembic's global context raised `KeyError('config')`, so jobs failed. Fixed with a lock in `init_db()`; regression test `test_init_db_is_safe_from_many_threads`. The integration test also leaked DB sessions while polling; fixed.

**Final Phase 6 test numbers:** SQLite + real Redis/Celery/moto-S3: 92 passed, 5 skipped. Real PostgreSQL 16 + pgvector (no integration folder): 91 passed. ruff clean.

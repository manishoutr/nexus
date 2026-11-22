# Environment and deployment

All settings are environment variables read in `app/config.py`. `.env.example` lists the common ones.

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | development | `production` refuses to start if `JWT_SECRET` is missing, short (<32) or a known placeholder |
| `JWT_SECRET` | none | Required in production. Dev generates a random per-process secret (tokens die on restart) |
| `DATABASE_URL` | sqlite:///./nexus.db | PostgreSQL needs the `pgvector` extension; Alembic creates it |
| `QUEUE_BACKEND` | db | `db` = in-process poller; `celery` = Redis broker + worker + beat |
| `REDIS_URL` | redis://localhost:6379/0 | Broker and result backend |
| `RATE_LIMIT_STORAGE` | memory:// | Use a Redis URL when running more than one API process |
| `STORAGE_BACKEND` | local | `s3` uses boto3; set `S3_ENDPOINT_URL` for an S3-compatible server (moto, Ceph, etc.), leave empty for AWS |
| `S3_BUCKET`, `AWS_REGION` | nexus-documents, us-east-1 | Credentials via the standard AWS variables |
| `ANTHROPIC_API_KEY`, `LLM_MODE` | none, auto | No key = offline extractive answers (the UI says so) |
| `EMBEDDING_BACKEND` | auto | `hash` (no ML deps), `st` (sentence-transformers, needs torch), `onnx` (same model via onnxruntime, no torch, `requirements-onnx.txt`), `auto` tries st, then onnx, then hash |
| `RERANK_ENABLED` | false | Cross-encoder reranker, needs sentence-transformers |
| `LOG_FORMAT`, `LOG_LEVEL` | text, INFO | `json` for log shippers |

## What a real deployment needs
- PostgreSQL with pgvector (the `pgvector/pgvector:pg16` image, or a host that offers the extension).
- Redis, if you use Celery.
- An S3 bucket (AWS) or an S3-compatible server, if you use S3 storage. Local disk is ephemeral on most free hosts.
- RAM, measured locally (peak resident memory of one process that loaded the embedder and embedded 40 texts):
  hashing 14 MB, ONNX MiniLM 203 MB, sentence-transformers MiniLM 476 MB (embedder alone, before the API, database
  driver and request handling). The API import alone is about 109 MB. So the torch path is not expected to fit
  Render's 512 MB free tier; the ONNX path might. **These are local measurements, not measurements on Render.**

## ONNX versus sentence-transformers
`scripts/onnx_parity.py` embeds 30 corpus paragraphs and 56 eval questions with both paths. Result: cosine 1.000000
(minimum and mean), largest component difference 2.1e-07, and the top-5 retrieval order was identical for 56 of 56
questions. The full evaluation with `--backend onnx` gives exactly the same report numbers as `--backend st`
(without the reranker). Because the vectors match, both backends record the same model name and can share an index.
The cross-encoder reranker still needs torch, so the ONNX path has no reranker and no reranker abstention gate; with
the ONNX path abstention accuracy is the 78.6% (test 75.0%) row of the evaluation table, not the 89.3% row.

## Docker
`docker-compose.yml` defines api, worker, beat, db, redis and moto as the S3 server (in-memory; the app creates the bucket itself).
`docker compose up --build` needs `JWT_SECRET` set; `WITH_ML=1` builds the sentence-transformers image.
**Status: the compose file was parsed as YAML and the Dockerfile reviewed, but Docker was not available
where this was written, so neither has been built or run.** The same service combination (real Redis, Celery
worker, S3-compatible moto, PostgreSQL 16 + pgvector) was tested as separate local processes; see `docs/UPGRADE_LOG.md`.

## CI
`.github/workflows/ci.yml`: ruff, tests on SQLite, tests on a pgvector Postgres service. Not yet run on GitHub.

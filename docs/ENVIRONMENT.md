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
| `STORAGE_BACKEND` | local | `s3` uses boto3; set `S3_ENDPOINT_URL` for MinIO, leave empty for AWS |
| `S3_BUCKET`, `AWS_REGION` | nexus-documents, us-east-1 | Credentials via the standard AWS variables |
| `ANTHROPIC_API_KEY`, `LLM_MODE` | none, auto | No key = offline extractive answers (the UI says so) |
| `EMBEDDING_BACKEND` | auto | `hash` (no ML deps), `st` (sentence-transformers), `auto` picks st if installed |
| `RERANK_ENABLED` | false | Cross-encoder reranker, needs sentence-transformers |
| `LOG_FORMAT`, `LOG_LEVEL` | text, INFO | `json` for log shippers |

## What a real deployment needs
- PostgreSQL with pgvector (the `pgvector/pgvector:pg16` image, or a host that offers the extension).
- Redis, if you use Celery.
- An S3 bucket (AWS) or MinIO, if you use S3 storage. Local disk is ephemeral on most free hosts.
- RAM: the hashing build is small. The sentence-transformers build pulls torch and needs roughly 1 GB or more
  (not measured here; render's free tier is 512 MB, so it is expected not to fit).

## Docker
`docker-compose.yml` defines api, worker, beat, db, redis, minio and a bucket-init job.
`docker compose up --build` needs `JWT_SECRET` set; `WITH_ML=1` builds the sentence-transformers image.
**Status: the compose file was parsed as YAML and the Dockerfile reviewed, but Docker was not available
where this was written, so neither has been built or run.** The same service combination (real Redis, Celery
worker, S3-compatible moto, PostgreSQL 16 + pgvector) was tested as separate local processes; see `docs/UPGRADE_LOG.md`.

## CI
`.github/workflows/ci.yml`: ruff, tests on SQLite, tests on a pgvector Postgres service. Not yet run on GitHub.

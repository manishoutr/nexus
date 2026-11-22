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

"""pgvector behaviour. Runs only against PostgreSQL (scripts/test_on_postgres.py)."""
import pytest

from app.db import SessionLocal, is_postgres, reset_db_for_tests

pytestmark = pytest.mark.skipif(not is_postgres(), reason="needs PostgreSQL + pgvector")

pytest.importorskip("sentence_transformers")

from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.models import Document, IngestionJob, User, Workspace  # noqa: E402
from app.services import embeddings, qa, vectorstore, worker  # noqa: E402
from app.services.storage import get_storage, new_key  # noqa: E402

DOCS = {
    "a.txt": b"The Odyssey card is a common payment card used on the Mo Bus public transport service.",
    "b.txt": b"Sambalpuri sarees are woven with the ikat technique in western Odisha.",
    "c.txt": b"Reciprocal rank fusion merges the rankings produced by keyword and vector retrieval.",
}


@pytest.fixture(autouse=True)
def st_backend(monkeypatch):
    monkeypatch.setattr(settings, "embedding_backend", "st")
    embeddings.reset_backend_for_tests()
    reset_db_for_tests()
    db = SessionLocal()
    db.add(User(email="o@example.com", password_hash="x")); db.flush()
    db.add(Workspace(name="w", owner_id=1)); db.commit()
    for name, data in DOCS.items():
        key = new_key(1); get_storage().put(key, data)
        d = Document(workspace_id=1, filename=name, storage_key=key, content_hash=name, status="pending")
        db.add(d); db.flush(); db.add(IngestionJob(document_id=d.id))
    db.commit(); db.close()
    while worker.run_job() is not None:
        pass
    yield
    embeddings.reset_backend_for_tests()


def test_extension_and_index_exist():
    db = SessionLocal()
    assert db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
    idx = db.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_chunks_embedding_vec_hnsw'")).scalar()
    assert idx and "hnsw" in idx and "vector_cosine_ops" in idx


def test_vectors_are_written_with_the_right_dimension():
    db = SessionLocal()
    n, bad = db.execute(text("SELECT count(*), count(*) FILTER (WHERE vector_dims(embedding_vec) <> 384 "
                             "OR embedding_vec IS NULL) FROM chunks")).one()
    assert n == 3 and bad == 0


def test_hnsw_index_can_serve_the_similarity_query():
    db = SessionLocal()
    db.execute(text("SET enable_seqscan = off"))
    plan = "\n".join(r[0] for r in db.execute(text(
        "EXPLAIN SELECT id FROM chunks ORDER BY embedding_vec <=> "
        "(SELECT embedding_vec FROM chunks LIMIT 1) LIMIT 3")))
    assert "ix_chunks_embedding_vec_hnsw" in plan


def test_retrieval_runs_through_pgvector_and_matches_python_cosine(monkeypatch):
    calls = {"n": 0}
    real = vectorstore.search

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(vectorstore, "search", spy)
    db = SessionLocal()
    pg = qa.retrieve(db, 1, "which card do I use to pay for buses?", 3)
    assert calls["n"] == 1 and "Odyssey" in pg[0].chunk.text
    # same ranking when pgvector is bypassed (pure-Python cosine over JSON vectors)
    monkeypatch.setattr(vectorstore, "usable", lambda *a: False)
    py = qa.retrieve(db, 1, "which card do I use to pay for buses?", 3)
    assert [r.chunk.id for r in pg] == [r.chunk.id for r in py]
    assert abs(pg[0].strength - py[0].strength) < 0.01


def test_workspace_isolation_in_vector_search():
    db = SessionLocal()
    db.add(Workspace(name="other", owner_id=1)); db.commit()
    assert qa.retrieve(db, 2, "Odyssey card", 3) == []
    sims = vectorstore.search(db, 2, embeddings.embed("Odyssey card"), embeddings.backend_name())
    assert sims == {}

"""pgvector access (PostgreSQL only).

On Postgres the 384-dim MiniLM vectors are also written to chunks.embedding_vec
and similarity search runs in the database (HNSW, cosine). On SQLite, or with
the 256-dim hashing fallback, nothing here is used and retrieval falls back to
the JSON vectors in Python. Raw SQL is used so the ORM models stay portable.
"""
from typing import Dict, Sequence

from sqlalchemy import text

from ..db import is_postgres

DIM = 384


def usable(model_name: str, dim: int) -> bool:
    return is_postgres() and dim == DIM and not model_name.startswith("hashing")


def _lit(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def store_vectors(db, chunk_ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
    if not is_postgres() or not chunk_ids:
        return
    db.execute(text("UPDATE chunks SET embedding_vec = CAST(:v AS vector) WHERE id = :id"),
               [{"v": _lit(v), "id": i} for i, v in zip(chunk_ids, vectors) if len(v) == DIM])


def search(db, workspace_id: int, query_vec: Sequence[float], model: str,
           limit: int = 50) -> Dict[int, float]:
    """Top `limit` chunk ids in the workspace by cosine similarity (1 = same)."""
    rows = db.execute(text(
        "SELECT id, 1 - (embedding_vec <=> CAST(:q AS vector)) AS sim FROM chunks "
        "WHERE workspace_id = :w AND embedding_model = :m AND embedding_vec IS NOT NULL "
        "ORDER BY embedding_vec <=> CAST(:q AS vector) LIMIT :n"),
        {"q": _lit(query_vec), "w": workspace_id, "m": model, "n": limit}).all()
    return {int(r[0]): float(r[1]) for r in rows}

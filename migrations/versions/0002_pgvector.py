"""pgvector column and HNSW index (PostgreSQL only; no-op on SQLite).

chunks.embedding_vec holds the 384-dim sentence-transformers vector so
similarity search runs inside Postgres. The JSON `embedding` column stays: it
is what the hashing fallback and SQLite use.

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_vec vector(384)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_embedding_vec_hnsw "
               "ON chunks USING hnsw (embedding_vec vector_cosine_ops)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_vec_hnsw")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_vec")

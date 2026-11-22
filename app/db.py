"""Database setup. SQLite by default; PostgreSQL via DATABASE_URL.
JSON blobs are stored as Text so the same code runs on both engines."""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    from . import models  # noqa: F401  (register models)
    Base.metadata.create_all(engine)
    _ensure_columns(engine)


# Columns added after the first release, as (table, column, DDL type).
# create_all never alters existing tables, so existing databases get these
# columns here. This shim is deliberately small; Alembic (see Phase 6 of the
# upgrade) takes over real migrations.
_LATE_COLUMNS = [
    ("chunks", "embedding_model", "VARCHAR DEFAULT ''"),
    ("workspaces", "owner_id", "INTEGER"),
    ("workspaces", "is_public", "INTEGER DEFAULT 0"),
    ("documents", "storage_key", "VARCHAR DEFAULT ''"),
    ("documents", "content_hash", "VARCHAR"),
    ("ingestion_jobs", "next_run_at", "TIMESTAMP"),
    ("ingestion_jobs", "locked_at", "TIMESTAMP"),
    ("ingestion_jobs", "worker_id", "VARCHAR DEFAULT ''"),
]
_LATE_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_ws_hash ON documents (workspace_id, content_hash)",
]


def _ensure_columns(eng) -> None:
    from sqlalchemy import inspect, text
    insp = inspect(eng)
    with eng.begin() as conn:
        for table, column, ddl in _LATE_COLUMNS:
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        for ddl in _LATE_INDEXES:
            conn.execute(text(ddl))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""Alembic: fresh schema matches the models; a pre-Alembic database is adopted."""
import tempfile
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext

from app.db import Base

ROOT = Path(__file__).resolve().parent.parent


def _cfg(conn):
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["connection"] = conn
    return cfg


def _sqlite():
    return sa.create_engine("sqlite:///" + str(Path(tempfile.mkdtemp()) / "m.db"))


def test_fresh_migration_matches_models_exactly():
    eng = _sqlite()
    with eng.begin() as conn:
        command.upgrade(_cfg(conn), "head")
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations disagree: {diff}"


def test_upgrade_is_repeatable_and_downgrade_works():
    eng = _sqlite()
    with eng.begin() as conn:
        cfg = _cfg(conn)
        command.upgrade(cfg, "head")
        command.upgrade(cfg, "head")        # no-op the second time
        command.downgrade(cfg, "base")
        assert sa.inspect(conn).get_table_names() == ["alembic_version"]


def test_database_created_before_alembic_is_adopted_without_data_loss():
    """The old create_all schema (no storage_key, no job lock columns, no
    users) with a row in it must upgrade in place."""
    eng = _sqlite()
    with eng.begin() as conn:
        conn.execute(sa.text("CREATE TABLE workspaces (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, created_at DATETIME)"))
        conn.execute(sa.text("CREATE TABLE documents (id INTEGER PRIMARY KEY, workspace_id INTEGER, filename VARCHAR NOT NULL, "
                             "content_type VARCHAR, size_bytes INTEGER, page_count INTEGER, status VARCHAR, error TEXT, created_at DATETIME)"))
        conn.execute(sa.text("CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, workspace_id INTEGER, seq INTEGER, "
                             "text TEXT, page INTEGER, section VARCHAR, embedding TEXT)"))
        conn.execute(sa.text("CREATE TABLE ingestion_jobs (id INTEGER PRIMARY KEY, document_id INTEGER, status VARCHAR, attempts INTEGER, "
                             "max_attempts INTEGER, error TEXT, created_at DATETIME, updated_at DATETIME)"))
        conn.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (1, 'old library')"))
        conn.execute(sa.text("INSERT INTO documents (id, workspace_id, filename, status) VALUES (1, 1, 'old.pdf', 'done')"))
        conn.execute(sa.text("INSERT INTO chunks (id, document_id, workspace_id, text) VALUES (1, 1, 1, 'kept')"))
        command.upgrade(_cfg(conn), "head")
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("documents")}
        assert {"storage_key", "content_hash"} <= cols
        assert {"embedding_model"} <= {c["name"] for c in insp.get_columns("chunks")}
        assert {"locked_at", "worker_id", "next_run_at"} <= {c["name"] for c in insp.get_columns("ingestion_jobs")}
        assert {"owner_id", "is_public"} <= {c["name"] for c in insp.get_columns("workspaces")}
        assert "users" in insp.get_table_names() and "memberships" in insp.get_table_names()
        assert conn.execute(sa.text("SELECT text FROM chunks")).scalar() == "kept"
        assert conn.execute(sa.text("SELECT name FROM workspaces")).scalar() == "old library"

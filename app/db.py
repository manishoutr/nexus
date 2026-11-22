"""Database setup. SQLite by default; PostgreSQL via DATABASE_URL.
JSON blobs are stored as Text so the same code runs on both engines."""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    """Bring the schema to the latest Alembic revision. Safe to call on every
    start: a current database is left alone, an old one is upgraded, and a
    database created before Alembic existed is adopted by the baseline."""
    from alembic import command
    from alembic.config import Config

    from . import models  # noqa: F401  (register models)
    cfg = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    cfg.set_main_option("script_location",
                        str(Path(__file__).resolve().parent.parent / "migrations"))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def is_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db_for_tests() -> None:
    """Drop every table, including Alembic's bookkeeping, then re-migrate."""
    from sqlalchemy import text

    from . import models  # noqa: F401
    if is_postgres():
        # Tests leave sessions open; their locks would block DROP TABLE forever.
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").execute(text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"))
        engine.dispose()
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    init_db()

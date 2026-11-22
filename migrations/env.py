"""Alembic environment. The URL is taken from the app settings so the same
DATABASE_URL drives the app, the worker and migrations."""
from alembic import context
from sqlalchemy import create_engine, pool

from app import models  # noqa: F401  (register tables on Base.metadata)
from app.config import settings
from app.db import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Programmatic callers (tests, app startup) may pass a connection instead.
    return config.get_main_option("sqlalchemy.url") or settings.database_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=_url().startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        engine = create_engine(_url(), poolclass=pool.NullPool)
        with engine.connect() as connection:
            _run(connection)
    else:
        _run(connection)


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=connection.dialect.name == "sqlite",
                      compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

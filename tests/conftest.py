"""Test environment, set before any app module is imported.

TEST_DATABASE_URL lets the same suite run against PostgreSQL; the default is
a throwaway SQLite file.
"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "sqlite:///" + os.path.join(_tmp, "test.db"))
os.environ["LLM_MODE"] = "offline"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["JOB_BACKOFF_BASE_S"] = "0.2"
os.environ["QUEUE_BACKEND"] = "db"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["REDIS_URL"] = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6391/0")

# Snapshot for subprocesses: other test modules mutate os.environ later.
TEST_ENV = {k: os.environ[k] for k in (
    "DATABASE_URL", "LLM_MODE", "EMBEDDING_BACKEND", "DATA_DIR",
    "JOB_BACKOFF_BASE_S", "STORAGE_BACKEND", "REDIS_URL")}

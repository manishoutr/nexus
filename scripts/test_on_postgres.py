"""Run the test suite against a real PostgreSQL 16 + pgvector server.

Starts a throwaway server with the `pgserver` package (no Docker needed):
    pip install pgserver psycopg2-binary
    python scripts/test_on_postgres.py [pytest args]
"""
import os
import subprocess
import sys
import tempfile

import pgserver

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
srv = pgserver.get_server(tempfile.mkdtemp(prefix="nexus-pg-"))
srv.psql("CREATE DATABASE nexus_test;")
base = srv.get_uri()  # e.g. postgresql://postgres:@/postgres?host=/tmp/...
uri = base.replace("postgresql://", "postgresql+psycopg2://", 1).replace("/postgres?", "/nexus_test?", 1)
print("using", uri)
env = dict(os.environ, TEST_DATABASE_URL=uri)
args = sys.argv[1:] or ["tests", "-q", "-p", "no:warnings"]
sys.exit(subprocess.call([sys.executable, "-m", "pytest", *args], cwd=root, env=env))

"""End-to-end API flow, offline mode: workspace -> upload -> ingest -> search -> ask."""
import os
import tempfile
import time

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "api.db")
os.environ["LLM_MODE"] = "offline"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _login(email="api@example.com"):
    r = client.post("/api/auth/register", json={"email": email, "password": "correct horse"})
    if r.status_code == 409:
        r = client.post("/api/auth/login", json={"email": email, "password": "correct horse"})
    client.headers["Authorization"] = "Bearer " + r.json()["access_token"]

DOC = b"""Retrieval Notes

Hybrid retrieval combines BM25 keyword ranking with vector similarity.
BM25 is precise for rare terms and names. Vector search tolerates rewording.
A grounded system cites the passages it used and abstains when evidence
is insufficient.
"""


def test_full_demo_flow():
    with client:  # startup: tables + background worker
        _login()
        # 1. workspace
        res = client.post("/api/workspaces", json={"name": "demo"})
        assert res.status_code == 200
        ws_id = res.json()["id"]

        # 2. upload
        res = client.post(f"/api/workspaces/{ws_id}/documents",
                          files={"file": ("notes.txt", DOC, "text/plain")})
        assert res.status_code == 201, res.text
        doc_id = res.json()["id"]
        assert res.json()["status"] == "pending"

        # 3. background worker processes it
        status = ""
        for _ in range(40):
            status = client.get(f"/api/documents/{doc_id}").json()["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.25)
        assert status == "done"

        # 4. hybrid search returns the passage with document reference
        res = client.post(f"/api/workspaces/{ws_id}/search",
                          json={"query": "hybrid retrieval BM25", "k": 3})
        assert res.status_code == 200
        data = res.json()
        assert data["count"] > 0
        assert data["results"][0]["document_name"] == "notes.txt"

        # 5. grounded ask with citations
        res = client.post(f"/api/workspaces/{ws_id}/ask",
                          json={"question": "What is hybrid retrieval?"})
        assert res.status_code == 200
        answer = res.json()
        assert not answer["abstained"]
        assert answer["citations"]
        assert answer["citations"][0]["document_name"] == "notes.txt"

        # 6. abstention on an out-of-scope question
        res = client.post(f"/api/workspaces/{ws_id}/ask",
                          json={"question": "Who won the 1998 football world cup final?"})
        assert res.json()["abstained"]


def test_upload_validation():
    with client:
        _login()
        ws_id = client.post("/api/workspaces", json={"name": "validation"}).json()["id"]
        res = client.post(f"/api/workspaces/{ws_id}/documents",
                          files={"file": ("evil.exe", b"nope", "application/octet-stream")})
        assert res.status_code == 415
        res = client.post(f"/api/workspaces/{ws_id}/documents",
                          files={"file": ("empty.txt", b"", "text/plain")})
        assert res.status_code == 422
        res = client.post("/api/workspaces/9999/documents",
                          files={"file": ("a.txt", b"hi", "text/plain")})
        assert res.status_code == 404


def test_metrics_endpoint():
    with client:
        _login()
        client.get("/api/health")
        res = client.get("/api/metrics")
        assert res.status_code == 200
        body = res.json()
        assert body["requests"] >= 1
        assert "jobs" in body

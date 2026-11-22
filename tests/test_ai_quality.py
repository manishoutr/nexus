"""Phase 5: citation verification, LLM failure handling, abstention paths."""
import json
from types import SimpleNamespace

import pytest

from app.db import SessionLocal, init_db, reset_db_for_tests
from app.models import Chunk, Document, User, Workspace
from app.schemas import Citation
from app.services import llm as llm_mod
from app.services import qa, worker
from app.services.citations import (strip_unverified_inline_refs,
                                    verify_citation, verify_citations)
from app.services.llm import LLMError, LLMService
from app.services.storage import get_storage, new_key

DOC = (b"Reciprocal rank fusion merges two ranked lists into one order. "
       b"Hybrid retrieval combines BM25 keyword ranking with vector similarity. "
       b"A grounded system cites its sources and abstains when evidence is missing.")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    worker.stop_worker()
    reset_db_for_tests()
    init_db()
    monkeypatch.setattr(llm_mod, "RETRY_BASE_S", 0.0)
    db = SessionLocal()
    db.add(User(email="o@example.com", password_hash="x"))
    db.flush()
    for ws in (1, 2):
        db.add(Workspace(name=f"w{ws}", owner_id=1))
    db.commit()
    db.close()


def ingest(ws, name, data):
    db = SessionLocal()
    key = new_key(ws)
    get_storage().put(key, data)
    d = Document(workspace_id=ws, filename=name, storage_key=key,
                 content_hash=f"{ws}-{name}", status="pending")
    db.add(d); db.flush()
    from app.models import IngestionJob
    db.add(IngestionJob(document_id=d.id)); db.commit()
    assert worker.run_job() == "done"
    return d.id


def good_citation(db, doc_id):
    ch = db.query(Chunk).filter_by(document_id=doc_id).first()
    d = db.query(Document).get(doc_id)
    return Citation(chunk_id=ch.id, document_id=doc_id, document_name=d.filename,
                    page=ch.page, section=ch.section, excerpt=ch.text[:100])


# --- citation verification --------------------------------------------------

def test_valid_citation_passes_and_each_kind_of_bad_one_fails():
    doc = ingest(1, "notes.txt", DOC)
    other = ingest(2, "other.txt", DOC + b" extra")
    db = SessionLocal()
    c = good_citation(db, doc)
    assert verify_citation(db, 1, c) is None
    bad = {
        "missing chunk": c.model_copy(update={"chunk_id": 9999}),
        "other workspace": c,                                    # checked against ws 2 below
        "wrong document": c.model_copy(update={"document_id": other}),
        "wrong name": c.model_copy(update={"document_name": "invented.pdf"}),
        "wrong page": c.model_copy(update={"page": 42}),
        "fabricated excerpt": c.model_copy(update={"excerpt": "text that is not in the chunk"}),
    }
    for why, cit in bad.items():
        ws = 2 if why == "other workspace" else 1
        assert verify_citation(db, ws, cit) is not None, why
    valid, problems = verify_citations(db, 1, [c, bad["missing chunk"], c])
    assert len(valid) == 1 and len(problems) == 1     # bad one dropped, duplicate collapsed


def test_inline_references_are_checked_against_retrieved_passages():
    allowed = [("notes.pdf", 3)]
    text, removed = strip_unverified_inline_refs(
        "Fusion merges lists [notes.pdf, page 3] and was invented [fake.pdf, page 9] [other.txt].",
        allowed)
    assert "[notes.pdf, page 3]" in text and "fake.pdf" not in text and "other.txt" not in text
    assert len(removed) == 2
    plain, none = strip_unverified_inline_refs("Array access a[1] and [sic] stay.", allowed)
    assert "[sic]" in plain and none == []


# --- LLM failures ---------------------------------------------------------

class APITimeoutError(Exception):
    pass


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def client_with(*outcomes):
    """Fake SDK client: each call pops one outcome (exception = raise)."""
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        o = outcomes[min(calls["n"], len(outcomes)) - 1]
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=o)])

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def run_llm(client):
    from pydantic import BaseModel

    class M(BaseModel):
        answer: str = ""
        used_chunk_ids: list = []
    return LLMService(client=client, enabled=True).complete_json("s", "u", M)


def test_timeout_is_retried_then_raised_as_classified_error():
    client, calls = client_with(APITimeoutError("t"))
    with pytest.raises(LLMError) as e:
        run_llm(client)
    assert e.value.kind == "timeout" and calls["n"] == 3     # 1 try + 2 retries


def test_rate_limit_then_success():
    client, calls = client_with(RateLimitError("429"), json.dumps({"answer": "ok", "used_chunk_ids": [0]}))
    assert run_llm(client).answer == "ok" and calls["n"] == 2


def test_auth_error_not_retried_and_invalid_output_classified():
    client, calls = client_with(AuthenticationError("bad key"))
    with pytest.raises(LLMError) as e:
        run_llm(client)
    assert e.value.kind == "auth" and calls["n"] == 1
    client, calls = client_with("this is not json")
    with pytest.raises(LLMError) as e:
        run_llm(client)
    assert e.value.kind == "invalid_output" and calls["n"] == 1
    client, _ = client_with('```json\n{"answer": "fenced", "used_chunk_ids": []}\n```')
    assert run_llm(client).answer == "fenced"


def use_llm(monkeypatch, client):
    monkeypatch.setattr(qa, "llm", LLMService(client=client, enabled=True))


def test_ask_falls_back_to_offline_when_claude_times_out(monkeypatch):
    ingest(1, "notes.txt", DOC)
    client, calls = client_with(APITimeoutError("t"))
    use_llm(monkeypatch, client)
    out = qa.ask(SessionLocal(), 1, "What does reciprocal rank fusion do?")
    assert out["mode"] == "offline" and not out["abstained"] and out["citations"]
    assert any("unavailable (timeout)" in w for w in out["warnings"])
    assert calls["n"] == 3


def test_claude_fabricated_reference_is_removed_and_citations_are_real(monkeypatch):
    ingest(1, "notes.txt", DOC)
    payload = json.dumps({"answer": "Fusion merges lists [invented.pdf, page 7].",
                          "used_chunk_ids": [0], "insufficient_evidence": False})
    use_llm(monkeypatch, client_with(payload)[0])
    db = SessionLocal()
    out = qa.ask(db, 1, "What does reciprocal rank fusion do?")
    assert out["mode"] == "claude" and "invented.pdf" not in out["answer"]
    assert any("Removed 1 reference" in w for w in out["warnings"])
    c = out["citations"][0]
    assert verify_citation(db, 1, c) is None and c.document_name == "notes.txt"


def test_claude_with_no_valid_citation_is_discarded(monkeypatch):
    ingest(1, "notes.txt", DOC)
    payload = json.dumps({"answer": "Made up.", "used_chunk_ids": [77, -3]})
    use_llm(monkeypatch, client_with(payload)[0])
    out = qa.ask(SessionLocal(), 1, "What does reciprocal rank fusion do?")
    assert out["mode"] == "offline" and "Made up" not in out["answer"]
    assert any("cited no retrieved passage" in w for w in out["warnings"])


def test_claude_can_abstain_itself(monkeypatch):
    ingest(1, "notes.txt", DOC)
    payload = json.dumps({"answer": "", "insufficient_evidence": True, "used_chunk_ids": []})
    use_llm(monkeypatch, client_with(payload)[0])
    out = qa.ask(SessionLocal(), 1, "What does reciprocal rank fusion do?")
    assert out["abstained"] and out["mode"] == "claude" and out["citations"] == []


def test_citation_that_disappears_is_dropped_and_answer_abstains(monkeypatch):
    doc = ingest(1, "notes.txt", DOC)
    real = qa._citation

    def ghost(r):
        c = real(r)
        return c.model_copy(update={"chunk_id": 424242})   # points at a chunk that isn't there

    monkeypatch.setattr(qa, "_citation", ghost)
    out = qa.ask(SessionLocal(), 1, "What does reciprocal rank fusion do?")
    assert out["abstained"] and out["citations"] == []
    assert any("does not exist" in w for w in out["warnings"])


def test_question_without_evidence_abstains():
    ingest(1, "notes.txt", DOC)
    out = qa.ask(SessionLocal(), 1, "Who won the 1998 football world cup final?")
    assert out["abstained"] and out["citations"] == []

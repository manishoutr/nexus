"""Q&A grounding: answers cite sources; weak evidence means abstention."""
import os
import tempfile

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "qa.db")
os.environ["LLM_MODE"] = "offline"

from app.db import init_db, SessionLocal  # noqa: E402
from app.models import Chunk, Document, Workspace  # noqa: E402
from app.services.embeddings import embed  # noqa: E402
from app.services.qa import ask  # noqa: E402
import json  # noqa: E402

init_db()


def _seed():
    db = SessionLocal()
    ws = Workspace(name="research")
    db.add(ws)
    db.flush()
    doc = Document(workspace_id=ws.id, filename="rag-notes.txt", status="done")
    db.add(doc)
    db.flush()
    db.add(Chunk(document_id=doc.id, workspace_id=ws.id, seq=0, page=1,
                 section="Retrieval",
                 text="Hybrid retrieval combines BM25 keyword search with "
                      "vector similarity. Keyword search is precise for names "
                      "and rare terms while vector search tolerates rewording.",
                 embedding=json.dumps(embed("Hybrid retrieval combines BM25 "
                                            "keyword search with vector similarity"))))
    db.commit()
    return db, ws.id


def test_relevant_question_gets_cited_answer():
    db, ws_id = _seed()
    result = ask(db, ws_id, "What is hybrid retrieval?")
    assert not result["abstained"]
    assert result["citations"]
    assert result["citations"][0].document_name == "rag-notes.txt"
    assert result["citations"][0].page == 1
    db.close()


def test_unanswerable_question_abstains():
    db, ws_id = _seed()
    result = ask(db, ws_id, "What is the airspeed velocity of an unladen swallow?")
    assert result["abstained"]
    assert result["citations"] == []
    assert "evidence" in result["answer"].lower()
    db.close()


def test_empty_workspace_abstains():
    db = SessionLocal()
    ws = Workspace(name="empty")
    db.add(ws)
    db.commit()
    result = ask(db, ws.id, "anything at all?")
    assert result["abstained"]
    db.close()

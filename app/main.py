"""Nexus API: workspaces, document upload, ingestion status, search, Q&A."""
import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, init_db
from .models import Chunk, Document, IngestionJob, QaLog, Workspace
from .schemas import (AskRequest, AskResponse, DocumentOut, SearchRequest,
                      SearchResponse, WorkspaceIn, WorkspaceOut)
from .services import qa as qa_service
from .services.extract import ALLOWED_EXTENSIONS
from .services.llm import llm
from .services.worker import start_worker, stop_worker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nexus")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Nexus", version="1.0.0",
              description="AI knowledge and research engine (learning project)")

METRICS = {"requests": 0, "errors": 0, "total_latency_ms": 0.0,
           "uploads": 0, "searches": 0, "questions": 0, "abstentions": 0,
           "ingestion_failures": 0}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    METRICS["requests"] += 1
    METRICS["total_latency_ms"] += (time.time() - start) * 1000
    if response.status_code >= 500:
        METRICS["errors"] += 1
    return response


@app.on_event("startup")
def startup() -> None:
    init_db()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    app.state.worker = start_worker()


@app.on_event("shutdown")
def shutdown() -> None:
    stop_worker()


def _doc_out(doc: Document) -> DocumentOut:
    return DocumentOut(id=doc.id, workspace_id=doc.workspace_id,
                       filename=doc.filename, size_bytes=doc.size_bytes,
                       page_count=doc.page_count, status=doc.status,
                       error=doc.error or "")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {"status": "ok", "llm_mode": llm.mode,
            "workspaces": db.query(Workspace).count(),
            "chunks": db.query(Chunk).count()}


@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db)):
    avg = (METRICS["total_latency_ms"] / METRICS["requests"]) if METRICS["requests"] else 0.0
    jobs = db.query(IngestionJob).all()
    return {**METRICS, "avg_latency_ms": round(avg, 2),
            "jobs": {"total": len(jobs),
                     "done": sum(1 for j in jobs if j.status == "done"),
                     "failed": sum(1 for j in jobs if j.status == "failed"),
                     "retried": sum(1 for j in jobs if j.attempts > 1)}}


# --- workspaces -------------------------------------------------------------

@app.post("/api/workspaces", response_model=WorkspaceOut)
def create_workspace(body: WorkspaceIn, db: Session = Depends(get_db)):
    if not body.name.strip():
        raise HTTPException(422, "Workspace name must not be empty")
    ws = Workspace(name=body.name.strip())
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return WorkspaceOut(id=ws.id, name=ws.name)


@app.get("/api/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(db: Session = Depends(get_db)):
    return [WorkspaceOut(id=w.id, name=w.name)
            for w in db.query(Workspace).order_by(Workspace.id)]


# --- documents --------------------------------------------------------------

@app.post("/api/workspaces/{workspace_id}/documents", response_model=DocumentOut, status_code=201)
async def upload_document(workspace_id: int, file: UploadFile = File(...),
                          db: Session = Depends(get_db)):
    METRICS["uploads"] += 1
    ws = db.query(Workspace).filter_by(id=workspace_id).first()
    if ws is None:
        raise HTTPException(404, f"Workspace {workspace_id} not found")
    filename = Path(file.filename or "document").name  # strip any path tricks
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(422, "Empty file")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")

    dest_dir = Path(settings.data_dir) / str(workspace_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / filename).write_bytes(content)

    doc = Document(workspace_id=workspace_id, filename=filename,
                   content_type=file.content_type or "", size_bytes=len(content),
                   status="pending")
    db.add(doc)
    db.flush()
    db.add(IngestionJob(document_id=doc.id))
    db.commit()
    db.refresh(doc)
    return _doc_out(doc)


@app.get("/api/workspaces/{workspace_id}/documents", response_model=list[DocumentOut])
def list_documents(workspace_id: int, db: Session = Depends(get_db)):
    return [_doc_out(d) for d in db.query(Document)
            .filter_by(workspace_id=workspace_id).order_by(Document.id)]


@app.get("/api/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(id=document_id).first()
    if doc is None:
        raise HTTPException(404, f"Document {document_id} not found")
    return _doc_out(doc)


# --- search & ask -------------------------------------------------------------

@app.post("/api/workspaces/{workspace_id}/search", response_model=SearchResponse)
def search(workspace_id: int, req: SearchRequest, db: Session = Depends(get_db)):
    METRICS["searches"] += 1
    ranked = qa_service.retrieve(db, workspace_id, req.query, req.k)
    return SearchResponse(
        query=req.query, count=len(ranked),
        results=[{"chunk_id": r.chunk.id, "document_id": r.chunk.document_id,
                  "document_name": r.document_name, "page": r.chunk.page,
                  "section": r.chunk.section, "text": r.chunk.text,
                  "score": round(r.score, 3)} for r in ranked])


@app.post("/api/workspaces/{workspace_id}/ask", response_model=AskResponse)
def ask(workspace_id: int, req: AskRequest, db: Session = Depends(get_db)):
    METRICS["questions"] += 1
    if not req.question.strip():
        raise HTTPException(422, "Question must not be empty")
    result = qa_service.ask(db, workspace_id, req.question, req.k)
    if result["abstained"]:
        METRICS["abstentions"] += 1
    db.add(QaLog(workspace_id=workspace_id, question=req.question,
                 answer=result["answer"], mode=result["mode"],
                 abstained=1 if result["abstained"] else 0,
                 citations=__import__("json").dumps(
                     [c.model_dump() for c in result["citations"]])))
    db.commit()
    return AskResponse(**result)


# --- frontend ---------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

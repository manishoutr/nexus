"""Nexus API: workspaces, document upload, ingestion status, search, Q&A."""
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import auth
from .config import settings
from .db import SessionLocal, get_db, init_db
from .logging_config import request_id_var, setup_logging
from .models import Chunk, Document, IngestionJob, Membership, QaLog, User, Workspace
from .ratelimit import limiter
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    LoginIn,
    MemberIn,
    RefreshIn,
    RegisterIn,
    SearchRequest,
    SearchResponse,
    TokenOut,
    UserOut,
    WorkspaceIn,
    WorkspaceOut,
)
from .services import qa as qa_service
from .services.llm import llm
from .services.safety import clean_question
from .services.storage import get_storage, new_key
from .services.validate import UploadRejected, validate_upload
from .services.worker import enqueue, jlog, pending_job_ids, start_worker, stop_worker

setup_logging()
log = logging.getLogger("nexus")
access_log = logging.getLogger("nexus.access")

STATIC_DIR = Path(__file__).resolve().parent / "static"
BASE_DIR = Path(__file__).resolve().parent.parent
SEED_DIR = BASE_DIR / "seed_docs"
# One file per demo document (seed_docs also keeps .txt twins of the PDF/DOCX
# for inspection; only the primary formats are ingested).
SEED_FILES = ["retrieval-augmented-generation.pdf", "odisha-handlooms.txt",
              "bhubaneswar-smart-city.docx"]

app = FastAPI(title="Nexus", version="1.0.0",
              description="AI knowledge and research engine (learning project)")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

METRICS = {"requests": 0, "errors": 0, "total_latency_ms": 0.0,
           "uploads": 0, "searches": 0, "questions": 0, "abstentions": 0,
           "ingestion_failures": 0}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    try:
        try:
            response = await call_next(request)
        except Exception:
            METRICS["requests"] += 1
            METRICS["errors"] += 1
            log.exception("unhandled error: %s %s", request.method, request.url.path)
            raise
        ms = (time.time() - start) * 1000
        METRICS["requests"] += 1
        METRICS["total_latency_ms"] += ms
        if response.status_code >= 500:
            METRICS["errors"] += 1
        response.headers["x-request-id"] = rid
        if request.url.path != "/api/health":   # liveness probes would flood the log
            access_log.info(json.dumps({"event": "request", "method": request.method,
                                        "path": request.url.path, "status": response.status_code,
                                        "ms": round(ms, 1)}))
        return response
    finally:
        request_id_var.reset(token)


def seed_demo_library() -> None:
    """First boot only: create the demo workspace and ingest seed_docs.

    Mirrors the upload endpoint (file into data_dir, Document row, ingestion
    job) so the worker processes them exactly like user uploads.
    """
    db = SessionLocal()
    try:
        if db.query(Workspace).count():
            return
        ws = Workspace(name="Research library", owner_id=None, is_public=1)
        db.add(ws)
        db.flush()
        storage = get_storage()
        seeded = 0
        for name in SEED_FILES:
            src = SEED_DIR / name
            if not src.exists():
                continue
            content = src.read_bytes()
            key = new_key(ws.id)
            storage.put(key, content)
            doc = Document(workspace_id=ws.id, filename=name, content_type="",
                           storage_key=key,
                           content_hash=hashlib.sha256(content).hexdigest(),
                           size_bytes=len(content), status="pending")
            db.add(doc)
            db.flush()
            db.add(IngestionJob(document_id=doc.id))
            seeded += 1
        db.commit()
        log.info("seeded demo library with %d documents", seeded)
    finally:
        db.close()


@app.on_event("startup")
def startup() -> None:
    init_db()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    seed_demo_library()
    if settings.queue_backend == "celery":
        # Workers are separate processes. Push anything left pending (e.g.
        # seeded above, or enqueued while the broker was down).
        db = SessionLocal()
        try:
            for jid in pending_job_ids(db):
                enqueue(jid)
        finally:
            db.close()
    else:
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


@app.get("/api/ready")
def ready(response: Response, db: Session = Depends(get_db)):
    """Readiness: can this instance serve traffic right now? 503 if any
    dependency check fails. (/api/health only says the process is alive.)"""
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
        cfg = Config(str(BASE_DIR / "alembic.ini"))
        cfg.set_main_option("script_location", str(BASE_DIR / "migrations"))
        head = ScriptDirectory.from_config(cfg).get_current_head()
        current = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
        checks["migrations"] = "ok" if current == head else f"behind (db={current}, head={head})"
    except Exception as e:  # noqa: BLE001
        checks.setdefault("database", f"error: {type(e).__name__}")
    try:
        st = get_storage()
        st.exists("_readiness_probe")
        checks["storage"] = f"ok ({st.name})"
    except Exception as e:  # noqa: BLE001
        checks["storage"] = f"error: {type(e).__name__}"
    if settings.queue_backend == "celery":
        try:
            import redis as redis_lib
            redis_lib.Redis.from_url(settings.redis_url, socket_timeout=2).ping()
            checks["redis"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["redis"] = f"error: {type(e).__name__}"
    good = all(v.startswith("ok") for v in checks.values())
    if not good:
        response.status_code = 503
    return {"status": "ready" if good else "not_ready", "checks": checks}


@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db), user: User = Depends(auth.require_user)):
    avg = (METRICS["total_latency_ms"] / METRICS["requests"]) if METRICS["requests"] else 0.0
    jobs = db.query(IngestionJob).all()
    return {**METRICS, "avg_latency_ms": round(avg, 2),
            "jobs": {"total": len(jobs),
                     "done": sum(1 for j in jobs if j.status == "done"),
                     "failed": sum(1 for j in jobs if j.status == "failed"),
                     "retried": sum(1 for j in jobs if j.attempts > 1)}}


# --- workspaces -------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenOut, status_code=201)
@limiter.limit(settings.rate_limit_auth)
def register(request: Request, body: RegisterIn, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(409, "That email is already registered")
    user = User(email=email, password_hash=auth.hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "That email is already registered")
    return TokenOut(access_token=auth.make_token(user.id, "access"),
                    refresh_token=auth.make_token(user.id, "refresh"))


@app.post("/api/auth/login", response_model=TokenOut)
@limiter.limit(settings.rate_limit_auth)
def login(request: Request, body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email.strip().lower()).first()
    # Same error for unknown email and wrong password.
    ok = auth.verify_password(body.password, user.password_hash if user else
                              auth.hash_password("x"))  # equalise timing
    if not user or not ok:
        raise HTTPException(401, "Wrong email or password")
    return TokenOut(access_token=auth.make_token(user.id, "access"),
                    refresh_token=auth.make_token(user.id, "refresh"))


@app.post("/api/auth/refresh", response_model=TokenOut)
@limiter.limit(settings.rate_limit_auth)
def refresh(request: Request, body: RefreshIn, db: Session = Depends(get_db)):
    uid = auth.decode_token(body.refresh_token, "refresh")
    if db.query(User).filter_by(id=uid).first() is None:
        raise HTTPException(401, "Unknown user")
    return TokenOut(access_token=auth.make_token(uid, "access"),
                    refresh_token=auth.make_token(uid, "refresh"))


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(auth.require_user)):
    return UserOut(id=user.id, email=user.email)


@app.post("/api/workspaces", response_model=WorkspaceOut)
def create_workspace(body: WorkspaceIn, db: Session = Depends(get_db),
                     user: User = Depends(auth.require_user)):
    if not body.name.strip():
        raise HTTPException(422, "Workspace name must not be empty")
    ws = Workspace(name=body.name.strip(), owner_id=user.id, is_public=0)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return WorkspaceOut(id=ws.id, name=ws.name, role="owner")


@app.get("/api/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(db: Session = Depends(get_db),
                    user: Optional[User] = Depends(auth.optional_user)):
    """Public workspaces plus the ones this user owns or belongs to."""
    out = []
    for w in db.query(Workspace).order_by(Workspace.id):
        role = auth.role_in(db, w, user)
        if role:
            out.append(WorkspaceOut(id=w.id, name=w.name, role=role,
                                    is_public=bool(w.is_public)))
    return out


@app.post("/api/workspaces/{workspace_id}/members", status_code=204)
def add_member(workspace_id: int, body: MemberIn, db: Session = Depends(get_db),
               user: User = Depends(auth.require_user)):
    auth.require_role(db, workspace_id, user, "owner")
    target = db.query(User).filter_by(email=body.email.strip().lower()).first()
    if target is None:
        raise HTTPException(404, "No user with that email")
    if target.id == user.id:
        raise HTTPException(422, "You already own this workspace")
    m = db.query(Membership).filter_by(workspace_id=workspace_id, user_id=target.id).first()
    if m:
        m.role = body.role
    else:
        db.add(Membership(workspace_id=workspace_id, user_id=target.id, role=body.role))
    db.commit()
    return Response(status_code=204)


# --- documents --------------------------------------------------------------

@app.post("/api/workspaces/{workspace_id}/documents", response_model=DocumentOut, status_code=201)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(request: Request, workspace_id: int, response: Response,
                          file: UploadFile = File(...),
                          db: Session = Depends(get_db),
                          user: Optional[User] = Depends(auth.optional_user)):
    METRICS["uploads"] += 1
    auth.require_role(db, workspace_id, user, "editor")
    filename = Path(file.filename or "document").name  # display name only
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        validate_upload(filename, content, settings.max_upload_mb * 1024 * 1024)
    except UploadRejected as e:
        raise HTTPException(e.status, e.message)

    digest = hashlib.sha256(content).hexdigest()
    existing = (db.query(Document)
                .filter_by(workspace_id=workspace_id, content_hash=digest).first())
    if existing is not None:  # idempotent: same bytes, same document
        response.status_code = 200
        return _doc_out(existing)

    key = new_key(workspace_id)
    get_storage().put(key, content)
    doc = Document(workspace_id=workspace_id, filename=filename,
                   content_type=file.content_type or "", size_bytes=len(content),
                   storage_key=key, content_hash=digest, status="pending")
    db.add(doc)
    try:
        db.flush()
        job = IngestionJob(document_id=doc.id, max_attempts=settings.job_max_attempts)
        db.add(job)
        db.commit()
    except IntegrityError:  # lost a race with a concurrent identical upload
        db.rollback()
        get_storage().delete(key)
        existing = (db.query(Document)
                    .filter_by(workspace_id=workspace_id, content_hash=digest).first())
        response.status_code = 200
        return _doc_out(existing)
    db.refresh(doc)
    jlog("document_uploaded", document=doc.id, workspace=workspace_id,
         bytes=len(content), key=key)
    enqueue(job.id)
    return _doc_out(doc)


@app.get("/api/workspaces/{workspace_id}/documents", response_model=list[DocumentOut])
def list_documents(workspace_id: int, db: Session = Depends(get_db),
                   user: Optional[User] = Depends(auth.optional_user)):
    auth.require_role(db, workspace_id, user, "viewer")
    return [_doc_out(d) for d in db.query(Document)
            .filter_by(workspace_id=workspace_id).order_by(Document.id)]


@app.get("/api/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db),
                 user: Optional[User] = Depends(auth.optional_user)):
    doc = db.query(Document).filter_by(id=document_id).first()
    if doc is None:
        raise HTTPException(404, f"Document {document_id} not found")
    auth.require_role(db, doc.workspace_id, user, "viewer")
    return _doc_out(doc)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: Session = Depends(get_db),
                    user: Optional[User] = Depends(auth.optional_user)):
    """Remove the document, its chunks, its jobs and the stored file."""
    doc = db.query(Document).filter_by(id=document_id).first()
    if doc is None:
        raise HTTPException(404, f"Document {document_id} not found")
    auth.require_role(db, doc.workspace_id, user, "editor")
    key = doc.storage_key or f"{doc.workspace_id}/{doc.filename}"
    db.query(Chunk).filter_by(document_id=doc.id).delete()
    db.query(IngestionJob).filter_by(document_id=doc.id).delete()
    db.delete(doc)
    db.commit()  # rows first: a leftover object is harmless, a leftover row is not
    try:
        get_storage().delete(key)
    except Exception:  # noqa: BLE001
        log.exception("could not delete stored object %s", key)
    jlog("document_deleted", document=document_id, key=key)
    return Response(status_code=204)


# --- search & ask -------------------------------------------------------------

@app.post("/api/workspaces/{workspace_id}/search", response_model=SearchResponse)
@limiter.limit(settings.rate_limit_ask)
def search(request: Request, workspace_id: int, req: SearchRequest,
           db: Session = Depends(get_db),
           user: Optional[User] = Depends(auth.optional_user)):
    auth.require_role(db, workspace_id, user, "viewer")
    METRICS["searches"] += 1
    ranked = qa_service.retrieve(db, workspace_id, req.query, req.k)
    return SearchResponse(
        query=req.query, count=len(ranked),
        results=[{"chunk_id": r.chunk.id, "document_id": r.chunk.document_id,
                  "document_name": r.document_name, "page": r.chunk.page,
                  "section": r.chunk.section, "text": r.chunk.text,
                  # Displayed score is the absolute evidence strength (0..1);
                  # ordering is by RRF rank, which is not human-meaningful.
                  "score": round(r.strength, 3)} for r in ranked])


@app.post("/api/workspaces/{workspace_id}/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit_ask)
def ask(request: Request, workspace_id: int, req: AskRequest,
        db: Session = Depends(get_db),
        user: Optional[User] = Depends(auth.optional_user)):
    auth.require_role(db, workspace_id, user, "viewer")
    METRICS["questions"] += 1
    question = clean_question(req.question)   # NUL etc. would also break PostgreSQL text columns
    if not question:
        raise HTTPException(422, "Question must not be empty")
    result = qa_service.ask(db, workspace_id, question, req.k)
    if result["abstained"]:
        METRICS["abstentions"] += 1
    db.add(QaLog(workspace_id=workspace_id, question=question,
                 answer=result["answer"], mode=result["mode"],
                 abstained=1 if result["abstained"] else 0,
                 citations=__import__("json").dumps(
                     [c.model_dump() for c in result["citations"]])))
    db.commit()
    return AskResponse(**result)


# --- frontend ---------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

"""ORM models. JSON fields are stored as JSON-encoded Text for portability."""
import json

from sqlalchemy import Column, DateTime, Integer, String, Text, func

from .db import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, default="")
    size_bytes = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    status = Column(String, default="pending", index=True)  # pending/processing/done/failed
    error = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, index=True)
    workspace_id = Column(Integer, index=True)
    seq = Column(Integer, default=0)
    text = Column(Text, default="")
    page = Column(Integer, default=0)      # 1-based for PDFs, 0 when unknown
    section = Column(String, default="")
    embedding = Column(Text, default="[]")  # JSON list of floats

    def embedding_list(self) -> list:
        return json.loads(self.embedding or "[]")


class IngestionJob(Base):
    """Durable job row: the worker polls this table, so restarts never lose
    work and failures are inspectable."""
    __tablename__ = "ingestion_jobs"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, index=True)
    status = Column(String, default="pending", index=True)  # pending/processing/done/failed
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    error = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class QaLog(Base):
    """Every question and answer, for auditability and evaluation."""
    __tablename__ = "qa_logs"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(Integer, index=True)
    question = Column(Text)
    answer = Column(Text)
    citations = Column(Text, default="[]")
    mode = Column(String, default="offline")
    abstained = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

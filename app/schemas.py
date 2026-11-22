"""Pydantic schemas: API contracts."""
from typing import List, Optional

from pydantic import BaseModel, Field


class WorkspaceIn(BaseModel):
    name: str


class WorkspaceOut(BaseModel):
    id: int
    name: str


class DocumentOut(BaseModel):
    id: int
    workspace_id: int
    filename: str
    size_bytes: int
    page_count: int
    status: str
    error: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class Passage(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    page: int
    section: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[Passage]


class AskRequest(BaseModel):
    question: str
    k: int = 5


class Citation(BaseModel):
    document_id: int
    document_name: str
    page: int
    section: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    mode: str
    abstained: bool

"""Pydantic schemas: API contracts."""
from typing import List, Optional

from pydantic import BaseModel, Field


class WorkspaceIn(BaseModel):
    name: str = Field(max_length=100)


class RegisterIn(BaseModel):
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class RefreshIn(BaseModel):
    refresh_token: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str


class MemberIn(BaseModel):
    email: str
    role: str = Field(pattern="^(viewer|editor)$")


class WorkspaceOut(BaseModel):
    id: int
    name: str
    role: str = ""
    is_public: bool = False


class DocumentOut(BaseModel):
    id: int
    workspace_id: int
    filename: str
    size_bytes: int
    page_count: int
    status: str
    error: str


class SearchRequest(BaseModel):
    query: str = Field(max_length=1000)
    k: Optional[int] = Field(default=None, ge=1, le=20)


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
    question: str = Field(max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: int
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
    warnings: List[str] = Field(default_factory=list)  # e.g. Claude failed, fell back
    flagged_passages: int = 0   # retrieved passages that looked like injected instructions

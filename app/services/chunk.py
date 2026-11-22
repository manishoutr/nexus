"""Chunking: split page text into overlapping chunks with metadata.

Strategy: sentence-aware split targeting ~600 characters with ~120 characters
of overlap, so a fact that straddles a boundary still appears whole in at
least one chunk. Each chunk keeps its page number and the nearest heading
seen above it (the "section"), which is what citations display.
"""
import re
from typing import List, Tuple

TARGET = 600
OVERLAP = 120

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADING_RE = re.compile(r"^([A-Z][A-Za-z0-9 &/\-]{2,60}):?\s*$")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_page(page: int, text: str) -> List[dict]:
    """Split one page's text into chunks. Returns dicts with text/page/section."""
    chunks: List[dict] = []
    current = ""
    section = ""
    for sent in _sentences(text):
        if _HEADING_RE.match(sent):
            section = _HEADING_RE.match(sent).group(1).strip()
        if current and len(current) + len(sent) > TARGET:
            chunks.append({"page": page, "section": section, "text": current.strip()})
            # overlap: carry the tail of the previous chunk forward
            current = current[-OVERLAP:] + " " + sent
        else:
            current = (current + " " + sent).strip()
    if current.strip():
        chunks.append({"page": page, "section": section, "text": current.strip()})
    return chunks


def chunk_document(pages: List[Tuple[int, str]]) -> List[dict]:
    out: List[dict] = []
    for page, text in pages:
        out.extend(chunk_page(page, text))
    for i, chunk in enumerate(out):
        chunk["seq"] = i
    return out

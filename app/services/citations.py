"""Citation verification.

A citation is only shown if it can be checked against the database:
  * the chunk exists and belongs to the same workspace,
  * the chunk belongs to the document the citation names,
  * the page matches the chunk's page metadata,
  * the document name matches the document row,
  * the excerpt really is text from that chunk.
Inline references like "[notes.txt, page 3]" inside an answer are checked
against the retrieved passages too, so the model cannot invent a source.
"""
import re
from typing import List, Optional, Tuple

from ..models import Chunk, Document

_INLINE = re.compile(r"\[([^\[\]]{1,120}?)(?:,\s*page\s*(\d+))?\]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_citation(db, workspace_id: int, cit) -> Optional[str]:
    """Return None if the citation checks out, else the reason it doesn't."""
    chunk = db.query(Chunk).filter_by(id=cit.chunk_id).first()
    if chunk is None:
        return f"chunk {cit.chunk_id} does not exist"
    if chunk.workspace_id != workspace_id:
        return f"chunk {cit.chunk_id} is not in this workspace"
    if chunk.document_id != cit.document_id:
        return f"chunk {cit.chunk_id} does not belong to document {cit.document_id}"
    doc = db.query(Document).filter_by(id=cit.document_id).first()
    if doc is None or doc.workspace_id != workspace_id:
        return f"document {cit.document_id} not found in this workspace"
    if _norm(doc.filename) != _norm(cit.document_name):
        return f"document name '{cit.document_name}' does not match document {cit.document_id}"
    if cit.page != chunk.page:
        return f"page {cit.page} does not match chunk {cit.chunk_id} (page {chunk.page})"
    if _norm(cit.excerpt) not in _norm(chunk.text):
        return f"excerpt is not text from chunk {cit.chunk_id}"
    return None


def verify_citations(db, workspace_id: int, citations) -> Tuple[list, List[str]]:
    """Split citations into (valid, reasons-for-dropped)."""
    ok, problems = [], []
    seen = set()
    for c in citations:
        why = verify_citation(db, workspace_id, c)
        if why:
            problems.append(why)
        elif c.chunk_id not in seen:
            seen.add(c.chunk_id)
            ok.append(c)
    return ok, problems


def strip_unverified_inline_refs(answer: str, allowed: List[Tuple[str, int]]) -> Tuple[str, List[str]]:
    """Remove "[name, page N]" references that don't match a retrieved
    (document name, page). allowed is a list of (filename, page)."""
    names = {_norm(n) for n, _ in allowed}
    pairs = {(_norm(n), p) for n, p in allowed}
    removed: List[str] = []

    def fix(m: "re.Match") -> str:
        name, page = m.group(1).strip(), m.group(2)
        # Only treat bracket text as a reference if it looks like a file name
        # or carries a page number; plain "[1]" or "[sic]" is left alone.
        looks_like_ref = bool(page) or re.search(r"\.\w{2,4}$", name)
        if not looks_like_ref:
            return m.group(0)
        n = _norm(name)
        if page:
            good = (n, int(page)) in pairs
        else:
            good = n in names
        if good:
            return m.group(0)
        removed.append(m.group(0))
        return ""

    return re.sub(r"\s{2,}", " ", _INLINE.sub(fix, answer)).strip(), removed

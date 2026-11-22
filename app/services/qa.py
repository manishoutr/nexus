"""Grounded question answering.

Pipeline: retrieve top-k chunks -> decide whether the evidence is strong
enough -> answer only from those chunks -> attach citations.

Abstention is a feature, not a failure: when the best chunk scores below the
threshold, Nexus says it does not have enough evidence instead of guessing.
The threshold is a documented constant (config.abstain_threshold).
"""
import re
from typing import List, Optional

from pydantic import BaseModel

from ..config import settings
from ..models import Chunk
from ..schemas import Citation
from .embeddings import tokenize
from .llm import llm
from .retrieval import hybrid_scores


class RetrievedChunk:
    def __init__(self, chunk: Chunk, document_name: str, score: float):
        self.chunk = chunk
        self.document_name = document_name
        self.score = score


def retrieve(db, workspace_id: int, query: str, k: int = 5) -> List[RetrievedChunk]:
    from ..models import Document
    chunks = db.query(Chunk).filter_by(workspace_id=workspace_id).all()
    if not chunks:
        return []
    doc_names = {d.id: d.filename
                 for d in db.query(Document).filter_by(workspace_id=workspace_id)}
    combined, _ = hybrid_scores([c.text for c in chunks],
                                [c.embedding_list() for c in chunks], query)
    ranked = sorted((RetrievedChunk(c, doc_names.get(c.document_id, "?"), s)
                     for c, s in zip(chunks, combined)),
                    key=lambda r: r.score, reverse=True)
    return ranked[: max(1, min(k, 20))]


def _citation(r: RetrievedChunk) -> Citation:
    return Citation(document_id=r.chunk.document_id,
                    document_name=r.document_name,
                    page=r.chunk.page, section=r.chunk.section,
                    excerpt=r.chunk.text[:200])


class _AnswerModel(BaseModel):
    answer: str
    used_chunk_ids: List[int]


_QA_SYSTEM = (
    "You are Nexus, a document Q&A engine. Answer ONLY from the provided "
    "passages. If the passages do not contain the answer, say the evidence "
    "is insufficient. Never use outside knowledge. Return JSON with keys: "
    "answer (string), used_chunk_ids (list of the chunk numbers you used).")


def _offline_answer(question: str, ranked: List[RetrievedChunk]) -> str:
    """Deterministic grounded answer: the sentences from the best chunks that
    share the most content words with the question, quoted with references."""
    q_terms = set(tokenize(question))
    best_sentences = []
    for r in ranked[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", r.chunk.text):
            overlap = len(q_terms & set(tokenize(sent)))
            if overlap > 0:
                best_sentences.append((overlap, r, sent.strip()))
    best_sentences.sort(key=lambda t: t[0], reverse=True)
    if not best_sentences:
        return ("The retrieved passages mention related terms but no sentence "
                "directly answers the question. See the citations to judge yourself.")
    lines = []
    for _score, r, sent in best_sentences[:3]:
        ref = f"[{r.document_name}" + (f", page {r.chunk.page}" if r.chunk.page else "") + "]"
        lines.append(f"{sent} {ref}")
    return "Based on the retrieved passages:\n\n" + "\n\n".join(lines)


def ask(db, workspace_id: int, question: str, k: int = 5) -> dict:
    """Returns {answer, citations, mode, abstained}."""
    ranked = retrieve(db, workspace_id, question, k)
    # Offline embeddings are lexical (hashing), so a top score built only on
    # vector signal is hashing noise: with enough chunks the max random
    # cosine clears the threshold on its own. Require at least one shared
    # content word before trusting an offline answer. Claude mode judges
    # relevance from the passages itself, so the gate stays score-only there.
    q_terms = set(tokenize(question))
    has_lexical = any(q_terms & set(tokenize(r.chunk.text)) for r in ranked[:3])
    weak = not ranked or ranked[0].score < settings.abstain_threshold
    if weak or (not llm.enabled and not has_lexical):
        return {"answer": "I don't have enough evidence in the uploaded "
                          "documents to answer that. Try adding relevant "
                          "documents or rephrasing the question.",
                "citations": [], "mode": llm.mode, "abstained": True}

    citations = [_citation(r) for r in ranked]

    if llm.enabled:
        passages = "\n\n".join(
            f"[chunk {i}] ({r.document_name}, page {r.chunk.page})\n{r.chunk.text}"
            for i, r in enumerate(ranked))
        result = llm.complete_json(
            _QA_SYSTEM,
            f"QUESTION: {question}\n\nPASSAGES:\n{passages}",
            _AnswerModel)
        if result is not None and result.answer.strip():
            used = set(result.used_chunk_ids)
            cited = [c for i, c in enumerate(citations) if i in used] or citations[:2]
            return {"answer": result.answer.strip(), "citations": cited,
                    "mode": "claude", "abstained": False}

    return {"answer": _offline_answer(question, ranked),
            "citations": citations[:3], "mode": "offline", "abstained": False}

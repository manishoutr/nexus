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
from . import vectorstore
from .embeddings import HASH_BACKEND_NAME, backend_name, embed, tokenize
from .citations import strip_unverified_inline_refs, verify_citations
from .llm import LLMError, llm
from .rerank import rerank
from .safety import SYSTEM_PROMPT, scan, build_passages_block, clean_question
from .retrieval import hybrid_scores


class RetrievedChunk:
    def __init__(self, chunk: Chunk, document_name: str, score: float,
                 strength: float = 0.0):
        self.chunk = chunk
        self.document_name = document_name
        self.score = score          # RRF score, used for ordering
        self.strength = strength    # absolute evidence signal, used to abstain


def retrieve(db, workspace_id: int, query: str, k: Optional[int] = None
             ) -> List[RetrievedChunk]:
    from ..models import Document
    if k is None:
        k = settings.retrieval_k
    chunks = db.query(Chunk).filter_by(workspace_id=workspace_id).all()
    if not chunks:
        return []
    doc_names = {d.id: d.filename
                 for d in db.query(Document).filter_by(workspace_id=workspace_id)}
    vector_scores = None
    qvec = embed(query)
    model = backend_name()
    if vectorstore.usable(model, len(qvec)):
        # Similarity search inside PostgreSQL (pgvector, HNSW index).
        sims = vectorstore.search(db, workspace_id, qvec, model, limit=max(50, k * 5))
        vector_scores = [sims.get(c.id, 0.0) for c in chunks]
    combined, parts = hybrid_scores([c.text for c in chunks],
                                    [] if vector_scores is not None
                                    else [c.embedding_list() for c in chunks],
                                    query, vector_scores=vector_scores)
    ranked = sorted((RetrievedChunk(c, doc_names.get(c.document_id, "?"), s, st)
                     for c, s, st in zip(chunks, combined, parts["strength"])),
                    key=lambda r: r.score, reverse=True)
    top = ranked[: max(1, min(k, 20))]
    return rerank(query, top)


def _citation(r: RetrievedChunk) -> Citation:
    return Citation(chunk_id=r.chunk.id, document_id=r.chunk.document_id,
                    document_name=r.document_name,
                    page=r.chunk.page, section=r.chunk.section,
                    excerpt=r.chunk.text[:200])


class _AnswerModel(BaseModel):
    answer: str = ""
    insufficient_evidence: bool = False
    used_chunk_ids: List[int] = []


def _offline_answer(question: str, ranked: List[RetrievedChunk]) -> str:
    """Deterministic grounded answer: the sentences from the best chunks that
    share the most content words with the question, quoted with references."""
    q_terms = set(tokenize(question))
    best_sentences = []
    for r in ranked[:3]:
        for sent in re.split(r"(?<=[.!?])\s+", r.chunk.text):
            if scan(sent):  # never echo injection-looking text as an answer
                continue
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


def _abstain_threshold() -> float:
    """The hashing fallback and sentence-transformers produce differently
    distributed scores, so each has its own calibrated threshold."""
    if backend_name() == HASH_BACKEND_NAME:
        return settings.abstain_threshold
    return settings.abstain_threshold_st


ABSTAIN_TEXT = ("I don't have enough evidence in the uploaded documents to "
                "answer that. Try adding relevant documents or rephrasing the question.")


def _abstain(mode: str, warnings=None, flagged: int = 0, text: str = ABSTAIN_TEXT) -> dict:
    return {"answer": text, "citations": [], "mode": mode, "abstained": True,
            "warnings": warnings or [], "flagged_passages": flagged}


def ask(db, workspace_id: int, question: str, k: Optional[int] = None) -> dict:
    """Returns {answer, citations, mode, abstained, warnings, flagged_passages}."""
    question = clean_question(question)
    ranked = retrieve(db, workspace_id, question, k)
    # With the hashing fallback the vector signal is lexical noise: with
    # enough chunks the max random cosine clears the threshold on its own.
    # Require at least one shared content word before trusting an offline
    # hashing answer. Sentence-transformer vectors are semantic, and Claude
    # mode judges relevance from the passages itself, so those gates stay
    # score-only.
    q_terms = set(tokenize(question))
    has_lexical = any(q_terms & set(tokenize(r.chunk.text)) for r in ranked[:3])
    weak = not ranked or ranked[0].strength < _abstain_threshold()
    lexical_gate = (not llm.enabled
                    and backend_name() == HASH_BACKEND_NAME and not has_lexical)
    scores = [r.rerank_score for r in ranked if hasattr(r, "rerank_score")]
    unanswerable_by_reranker = bool(scores) and max(scores) < settings.abstain_rerank_threshold
    if weak or lexical_gate or unanswerable_by_reranker:
        return _abstain(llm.mode)

    citations = [_citation(r) for r in ranked]
    flagged = sum(1 for r in ranked if scan(r.chunk.text))
    warnings: List[str] = []
    answer, cited, mode = None, [], "offline"

    if llm.enabled:
        passages, _ = build_passages_block(
            (i, r.document_name, r.chunk.page, r.chunk.text)
            for i, r in enumerate(ranked))
        try:
            result = llm.complete_json(
                SYSTEM_PROMPT,
                f"QUESTION: {question}\n\nPASSAGES:\n{passages}", _AnswerModel)
        except LLMError as e:
            warnings.append(f"Claude was unavailable ({e.kind}); showing the "
                            "extractive offline answer instead.")
            result = None
        if result is not None:
            used = sorted({i for i in result.used_chunk_ids if 0 <= i < len(ranked)})
            text, removed = strip_unverified_inline_refs(
                result.answer, [(r.document_name, r.chunk.page) for r in ranked])
            if removed:
                warnings.append(f"Removed {len(removed)} reference(s) in the answer "
                                "that do not match any retrieved passage.")
            if result.insufficient_evidence:
                return _abstain("claude", warnings, flagged)
            if used and text:
                answer, cited, mode = text[:4000], [citations[i] for i in used], "claude"
            else:
                warnings.append("Claude's answer cited no retrieved passage, so it "
                                "was discarded and the extractive answer is shown.")

    if answer is None:
        answer, cited = _offline_answer(question, ranked), citations[:3]

    # Last gate: nothing is shown unless it checks out against the database.
    valid, problems = verify_citations(db, workspace_id, cited)
    warnings.extend(f"Dropped a citation: {p}" for p in problems)
    if not valid:
        return _abstain(mode, warnings + ["No citation could be verified."], flagged)
    return {"answer": answer, "citations": valid, "mode": mode, "abstained": False,
            "warnings": warnings, "flagged_passages": flagged}

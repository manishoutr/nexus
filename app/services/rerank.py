"""Optional cross-encoder reranking stage.

Bi-encoder retrieval (BM25 + MiniLM vectors) is fast but scores each chunk
without seeing the query and the chunk together. A cross-encoder reads the
(query, chunk) pair jointly and is markedly better at ordering the top
candidates. It costs a second model download and CPU time per query, so it
is off by default (RERANK_ENABLED=true to enable) and only reorders the top
RERANK_TOP_N candidates - retrieval still does the recall work.
"""
import logging
import threading

from ..config import settings

log = logging.getLogger("nexus.rerank")

_lock = threading.Lock()
_cross_encoder = None
_unavailable_reason = None


def _load():
    global _cross_encoder, _unavailable_reason
    if _cross_encoder is not None or _unavailable_reason is not None:
        return _cross_encoder
    with _lock:
        if _cross_encoder is not None or _unavailable_reason is not None:
            return _cross_encoder
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(settings.rerank_model)
            log.info("cross-encoder reranker loaded: %s", settings.rerank_model)
        except Exception as e:  # noqa: BLE001 - missing dep, no network, OOM
            _unavailable_reason = str(e)
            log.warning("cross-encoder unavailable (%s); reranking disabled", e)
    return _cross_encoder


def rerank(query: str, ranked: list) -> list:
    """Reorder the top candidates by cross-encoder score.

    `ranked` is a list of objects with .chunk.text and .score (see
    qa.RetrievedChunk). The RRF order is preserved beyond RERANK_TOP_N and
    the returned objects keep their original .score (the RRF score stays the
    displayed/evidence score); only the order changes. If the reranker is
    disabled or unavailable, the input order is returned unchanged.
    """
    if not settings.rerank_enabled or len(ranked) < 2:
        return ranked
    ce = _load()
    if ce is None:
        return ranked
    head = ranked[: settings.rerank_top_n]
    pairs = [(query, r.chunk.text) for r in head]
    try:
        scores = ce.predict(pairs)
    except Exception as e:  # noqa: BLE001 - never let reranking break a query
        log.warning("cross-encoder scoring failed (%s); keeping RRF order", e)
        return ranked
    reordered = [r for _s, r in sorted(zip(scores, head),
                                       key=lambda t: t[0], reverse=True)]
    return reordered + ranked[settings.rerank_top_n:]


def reset_for_tests() -> None:
    global _cross_encoder, _unavailable_reason
    _cross_encoder = None
    _unavailable_reason = None

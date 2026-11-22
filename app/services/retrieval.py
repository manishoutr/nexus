"""Hybrid retrieval over chunks.

BM25 keyword retrieval and vector similarity retrieval run independently and
produce their own ranked lists; Reciprocal Rank Fusion (RRF) merges the two
rankings into the final order. Alongside the RRF order we compute an
absolute "strength" signal (not rank-based) which is what makes abstention
possible: a query with no real evidence stays near zero instead of being
stretched by normalization.
"""
import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from ..config import settings
from .embeddings import cosine, embed, tokenize


class BM25:
    """Minimal BM25 over an in-memory corpus (standard k1/b defaults)."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.n = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for toks, dl in zip(self.doc_tokens, self.doc_len):
            tf = Counter(toks)
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0))
                s += self._idf(t) * (f * (self.k1 + 1)) / (denom or 1.0)
            out.append(s)
        return out


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def bm25_rank(texts: Sequence[str], query: str) -> List[Tuple[int, float]]:
    """Independent keyword retrieval: (corpus index, BM25 score), best first."""
    raw = BM25(texts).scores(query)
    return sorted(enumerate(raw), key=lambda t: t[1], reverse=True)


def vector_rank(embeddings: Sequence[List[float]], query_vector: Sequence[float]
                ) -> List[Tuple[int, float]]:
    """Independent vector retrieval: (corpus index, cosine), best first."""
    raw = [cosine(query_vector, e) for e in embeddings]
    return sorted(enumerate(raw), key=lambda t: t[1], reverse=True)


def rrf_merge(rankings: Sequence[Sequence[Tuple[int, float]]], k: int = 60
              ) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score(i) = sum over rankings of 1/(k + rank_i).

    Rank-based, so a method that scores everything zero still contributes its
    ordering; items missing from a ranking simply get no points from it.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_scores(texts: Sequence[str], embeddings: Sequence[List[float]],
                  query: str, alpha: float = 0.5, vector_scores=None
                  ) -> Tuple[List[float], Dict[str, List[float]]]:
    """RRF-ordered combined scores plus per-method components.

    Returns (combined, components): combined[i] is the RRF score used for
    ranking; components carries the independent signals and an absolute
    "strength" per item (normalized keyword signal gated on a real term hit,
    plus raw-cosine-scaled vector signal) used by the abstention gate.
    """
    b_ranking = bm25_rank(texts, query)
    if vector_scores is not None:
        # Cosine similarities computed elsewhere (pgvector); chunks outside
        # the database's top-N get 0 and rank last.
        v_ranking = sorted(enumerate(vector_scores), key=lambda t: t[1], reverse=True)
    else:
        v_ranking = vector_rank(embeddings, embed(query))
    fused = rrf_merge([b_ranking, v_ranking], k=settings.rrf_k)
    n = len(texts)
    combined = [fused.get(i, 0.0) for i in range(n)]

    bm25_raw = [0.0] * n
    for i, s in b_ranking:
        bm25_raw[i] = s
    vec_raw = [0.0] * n
    for i, s in v_ranking:
        vec_raw[i] = s
    bm25 = _minmax(bm25_raw)
    vec = _minmax(vec_raw)
    strength = []
    for b, br, v, vr in zip(bm25, bm25_raw, vec, vec_raw):
        b_signal = b if br > 0 else 0.0
        v_signal = v * max(0.0, vr)
        strength.append(alpha * b_signal + (1 - alpha) * v_signal)
    return combined, {"bm25": bm25, "vector": vec,
                      "bm25_raw": bm25_raw, "vector_raw": vec_raw,
                      "strength": strength}

"""Hybrid retrieval over chunks: BM25 + vector cosine, min-max merged."""
import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

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


def hybrid_scores(texts: Sequence[str], embeddings: Sequence[List[float]],
                  query: str, alpha: float = 0.5) -> Tuple[List[float], Dict[str, List[float]]]:
    """Combined scores preserve ABSOLUTE evidence strength, which is what
    makes abstention possible: a query with no real matches scores near zero
    instead of being stretched to 0.5 by normalization. Keyword signal is
    zeroed when BM25 found no term match at all; vector signal is scaled by
    the raw cosine, so weak similarities stay weak."""
    bm25_raw = BM25(texts).scores(query)
    bm25 = _minmax(bm25_raw)
    qv = embed(query)
    vec_raw = [cosine(qv, e) for e in embeddings]
    vec = _minmax(vec_raw)
    combined = []
    for b, br, v, vr in zip(bm25, bm25_raw, vec, vec_raw):
        b_signal = b if br > 0 else 0.0
        v_signal = v * max(0.0, vr)
        combined.append(alpha * b_signal + (1 - alpha) * v_signal)
    return combined, {"bm25": bm25, "vector": vec,
                      "bm25_raw": bm25_raw, "vector_raw": vec_raw}

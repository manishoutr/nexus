"""Deterministic local text embeddings (same approach as CareerOS).

Sparse bag-of-words vectors via the hashing trick: deterministic, free, no
network. Trade-off: lexical overlap, not deep semantics - swap embed() for a
real embedding model and re-ingest to upgrade.
"""
import hashlib
import math
import re
from typing import List

DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

STOPWORDS = {"a", "an", "the", "in", "on", "at", "of", "to", "for",
             "and", "or", "with", "using", "is", "are", "by", "from",
             "as", "our", "you", "your", "we", "will", "this", "that",
             "it", "its", "be", "used", "use"}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed, naive plural folding."""
    tokens = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            tokens.append(tok[:-1])
    return tokens


def embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % DIM
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))

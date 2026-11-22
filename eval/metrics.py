"""Pure metric functions for the RAG evaluation. No I/O, easy to unit test.

Relevance is defined by evidence phrases, not chunk ids, so the numbers stay
valid if the chunking changes: a retrieved chunk is relevant to a question if
it contains one of that question's gold evidence phrases.
"""
import re
from typing import Dict, List, Sequence

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of and or to in on for with is are was were be by as at it its this that from "
            "what which who how when where why does do did can".split())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def contains(text: str, phrase: str) -> bool:
    return norm(phrase) in norm(text)


def relevant_flags(chunk_texts: Sequence[str], phrases: Sequence[str]) -> List[bool]:
    return [any(contains(t, p) for p in phrases) for t in chunk_texts]


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    top = list(flags)[:k]
    return sum(top) / k if k else 0.0


def recall_at_k(chunk_texts: Sequence[str], phrases: Sequence[str], k: int) -> float:
    """Share of the question's evidence items found in the top k chunks.
    Multi-document questions need every item to reach 1.0."""
    if not phrases:
        return 0.0
    top = chunk_texts[:k]
    return sum(any(contains(t, p) for t in top) for p in phrases) / len(phrases)


def reciprocal_rank(flags: Sequence[bool]) -> float:
    for i, f in enumerate(flags, start=1):
        if f:
            return 1.0 / i
    return 0.0


def content_tokens(s: str) -> List[str]:
    return [t for t in _WORD.findall(s.lower()) if t not in _STOP and len(t) > 1]


def sentence_support(sentence: str, sources: Sequence[str], threshold: float = 0.8) -> bool:
    """Heuristic: a sentence is supported if >= threshold of its content words
    occur in the cited text. Lexical, so it checks "is this said in the source",
    not "does it mean the same"."""
    toks = content_tokens(sentence)
    if not toks:
        return True
    pool = set(content_tokens(" ".join(sources)))
    return sum(t in pool for t in toks) / len(toks) >= threshold


def split_answer_sentences(answer: str) -> List[str]:
    body = re.sub(r"\[[^\]]*\]", " ", answer)                 # drop [source, page N]
    body = re.sub(r"^\s*based on the retrieved passages:?", " ", body, flags=re.I)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(content_tokens(s)) >= 2]


def faithfulness(answer: str, cited_texts: Sequence[str]) -> float:
    sents = split_answer_sentences(answer)
    if not sents:
        return 0.0
    return sum(sentence_support(s, cited_texts) for s in sents) / len(sents)


def answer_evidence_coverage(answer: str, phrases: Sequence[str], threshold: float = 0.7) -> float:
    """Share of evidence items whose key words appear in the answer."""
    if not phrases:
        return 0.0
    ans = set(content_tokens(answer))
    hit = 0
    for p in phrases:
        toks = content_tokens(p)
        if toks and sum(t in ans for t in toks) / len(toks) >= threshold:
            hit += 1
    return hit / len(phrases)


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def abstention_accuracy(rows: Sequence[Dict]) -> Dict[str, float]:
    """rows: dicts with answerable (bool) and abstained (bool)."""
    rows = list(rows)
    if not rows:
        return {"accuracy": 0.0, "false_abstain_rate": 0.0, "false_answer_rate": 0.0, "n": 0}
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    correct = sum(r["abstained"] != r["answerable"] for r in rows)
    return {"accuracy": round(correct / len(rows), 4),
            "false_abstain_rate": round(sum(r["abstained"] for r in ans) / len(ans), 4) if ans else 0.0,
            "false_answer_rate": round(sum(not r["abstained"] for r in unans) / len(unans), 4) if unans else 0.0,
            "n": len(rows)}

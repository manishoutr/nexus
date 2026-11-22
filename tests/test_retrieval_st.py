"""Sentence-transformers backend: real semantic embeddings.

Skipped when the model cannot be loaded (no network / no torch), so the
suite still passes in the light demo environment. Runs last alphabetically
and restores the hashing backend afterwards.
"""
import math

import pytest

pytest.importorskip("sentence_transformers")

from app.config import settings  # noqa: E402
from app.services import embeddings  # noqa: E402
from app.services.retrieval import hybrid_scores  # noqa: E402

CHUNKS = [
    "Hybrid retrieval combines keyword BM25 with vector similarity because "
    "keyword search is precise for names while vectors tolerate rewording.",
    "The Mo Bus network connects Bhubaneswar with Cuttack and Khordha. The "
    "Odyssey card is a common payment card for the service.",
    "Sambalpuri sarees are woven with the ikat technique where threads are "
    "tie-dyed before weaving. Motifs include the conch shell and wheel.",
]


@pytest.fixture(scope="module")
def st_backend():
    settings.embedding_backend = "st"
    embeddings.reset_backend_for_tests()
    try:
        backend = embeddings.get_backend()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"sentence-transformers model unavailable: {e}")
    yield backend
    settings.embedding_backend = "hash"
    embeddings.reset_backend_for_tests()


def test_model_metadata(st_backend):
    assert "MiniLM" in st_backend.name
    assert st_backend.dim == 384


def test_embeddings_are_normalized(st_backend):
    v = embeddings.embed("hybrid retrieval")
    assert len(v) == 384
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-4


def test_semantic_match_without_shared_keywords(st_backend):
    """A paraphrase with no content-word overlap must still find its chunk;
    the hashing fallback cannot do this."""
    texts = CHUNKS
    vecs = [embeddings.embed(c) for c in texts]
    _, parts = hybrid_scores(texts, vecs, "merging lexical and semantic search")
    assert parts["vector_raw"][0] > 0.3


def test_st_abstention_calibration(st_backend):
    """Documents the ABSTAIN_THRESHOLD_ST calibration on real vectors:
    an answerable paraphrase must clear it, an unrelated query must not."""
    vecs = [embeddings.embed(c) for c in CHUNKS]
    _, answerable = hybrid_scores(CHUNKS, vecs, "why combine keyword and vector search?")
    _, unrelated = hybrid_scores(CHUNKS, vecs, "who won the 1998 world cup final?")
    assert max(answerable["strength"]) > settings.abstain_threshold_st
    assert max(unrelated["strength"]) < settings.abstain_threshold_st

"""Retrieval quality: the supporting chunk must rank first for its question."""
from app.services.embeddings import embed
from app.services.retrieval import hybrid_scores

CHUNKS = [
    "Hybrid retrieval combines keyword BM25 with vector similarity because "
    "keyword search is precise for names while vectors tolerate rewording.",
    "The Mo Bus network connects Bhubaneswar with Cuttack and Khordha. The "
    "Odyssey card is a common payment card for the service.",
    "Sambalpuri sarees are woven with the ikat technique where threads are "
    "tie-dyed before weaving. Motifs include the conch shell and wheel.",
]
EMBEDDINGS = [embed(c) for c in CHUNKS]


def test_hybrid_finds_supporting_chunk():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS,
                                "why combine keyword and vector search?")
    assert combined[0] == max(combined)


def test_exact_name_query_uses_keyword_strength():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS, "Odyssey card")
    assert combined[1] == max(combined)


def test_unrelated_query_scores_low_everywhere():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS, "quantum chromodynamics")
    assert max(combined) < 0.5  # leaves room for the abstention threshold

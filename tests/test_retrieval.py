"""Retrieval: independent BM25/vector rankings, RRF fusion, abstention signal.

These run on the deterministic hashing backend (EMBEDDING_BACKEND=hash is set
by tests/test_api.py, collected first) so the suite needs no model download.
"""
from app.services.embeddings import HASH_BACKEND_NAME, backend_name, embed
from app.services.retrieval import bm25_rank, hybrid_scores, rrf_merge, vector_rank

CHUNKS = [
    "Hybrid retrieval combines keyword BM25 with vector similarity because "
    "keyword search is precise for names while vectors tolerate rewording.",
    "The Mo Bus network connects Bhubaneswar with Cuttack and Khordha. The "
    "Odyssey card is a common payment card for the service.",
    "Sambalpuri sarees are woven with the ikat technique where threads are "
    "tie-dyed before weaving. Motifs include the conch shell and wheel.",
]
EMBEDDINGS = [embed(c) for c in CHUNKS]


def test_backend_is_forced_hashing():
    assert backend_name() == HASH_BACKEND_NAME


def test_hybrid_finds_supporting_chunk():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS,
                                "why combine keyword and vector search?")
    assert combined[0] == max(combined)


def test_exact_name_query_uses_keyword_strength():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS, "Odyssey card")
    assert combined[1] == max(combined)


def test_unrelated_query_strength_low_everywhere():
    _, parts = hybrid_scores(CHUNKS, EMBEDDINGS, "quantum chromodynamics")
    assert max(parts["strength"]) < 0.5  # leaves room for the abstention threshold


def test_bm25_rank_is_independent_and_ordered():
    ranking = bm25_rank(CHUNKS, "Odyssey card")
    assert ranking[0][0] == 1            # the Mo Bus chunk wins on keywords
    scores = [s for _i, s in ranking]
    assert scores == sorted(scores, reverse=True)


def test_vector_rank_is_independent_and_ordered():
    ranking = vector_rank(EMBEDDINGS, embed("keyword and vector search"))
    assert ranking[0][0] == 0
    scores = [s for _i, s in ranking]
    assert scores == sorted(scores, reverse=True)


def test_rrf_merge_rewards_items_strong_in_both_lists():
    # item 0 tops both rankings; item 1 is second in one, absent from the other
    fused = rrf_merge([[(0, 9.0), (1, 1.0)], [(0, 0.9), (2, 0.1)]], k=60)
    assert fused[0] > fused[1]
    assert fused[0] > fused[2]
    # rank-based: raw score magnitudes must not leak through
    fused_small = rrf_merge([[(0, 0.001), (1, 0.0001)], [(0, 0.002), (2, 0.0001)]], k=60)
    assert fused_small[0] == fused[0]


def test_rrf_rank_drives_combined_order():
    combined, _ = hybrid_scores(CHUNKS, EMBEDDINGS, "Odyssey card")
    order = sorted(range(len(CHUNKS)), key=lambda i: combined[i], reverse=True)
    assert order[0] == 1

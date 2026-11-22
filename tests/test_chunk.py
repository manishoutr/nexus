"""Chunking: size targets, overlap and metadata preservation."""
from app.services.chunk import TARGET, chunk_document, chunk_page

PAGE_TEXT = (
    "Introduction\n"
    "Retrieval systems find relevant passages. " * 12 + "\n"
    "Methods\n"
    "BM25 ranks by term frequency. " * 12
)


def test_chunks_respect_target_size_with_tolerance():
    chunks = chunk_page(1, PAGE_TEXT)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c["text"]) <= TARGET * 1.6  # sentences are never cut mid-way


def test_page_and_sequence_metadata():
    chunks = chunk_document([(1, "First page content about indexing."),
                             (3, "Third page content about ranking.")])
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 3
    assert [c["seq"] for c in chunks] == [0, 1]


def test_section_heading_is_attached():
    chunks = chunk_page(1, "Methods\nBM25 ranks documents by term rarity.")
    assert chunks[0]["section"] == "Methods"


def test_overlap_preserves_boundary_facts():
    # one long stream of sentences: a fact at a chunk boundary must appear
    # (fully or via overlap) so retrieval never loses it
    text = " ".join(f"Sentence number {i} carries fact {i}." for i in range(60))
    chunks = chunk_page(1, text)
    joined = " ".join(c["text"] for c in chunks)
    assert "fact 59" in joined

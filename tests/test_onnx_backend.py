"""ONNX MiniLM backend: needs onnxruntime, tokenizers and the model download, so it skips without them."""
import math

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from app.services.embeddings import OnnxBackend  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def onnx():
    try:
        return OnnxBackend(MODEL)
    except Exception as e:  # noqa: BLE001 - no network / model not cached
        pytest.skip(f"model unavailable: {e}")


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_shape_and_unit_norm(onnx):
    v = onnx.embed_many(["hello world", "x " * 1000])  # second one is truncated, not an error
    assert len(v) == 2 and len(v[0]) == 384
    assert all(abs(math.sqrt(_dot(x, x)) - 1) < 1e-4 for x in v)


def test_paraphrase_beats_unrelated(onnx):
    a, b, c = onnx.embed_many(["How do I get my money back?", "Refunds are issued within 14 days.", "The Konark temple has stone wheels."])
    assert _dot(a, b) > _dot(a, c)


def test_matches_sentence_transformers(onnx):
    st = pytest.importorskip("sentence_transformers")
    texts = ["Hybrid retrieval merges two ranked lists.", "Odisha is known for ikat weaving.", "What is a presigned URL?"]
    ref = st.SentenceTransformer(MODEL).encode(texts, normalize_embeddings=True)
    got = onnx.embed_many(texts)
    for r, g in zip(ref, got):
        assert _dot(r.tolist(), g) > 0.9999

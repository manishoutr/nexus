"""Cross-encoder reranker, against the real model (skipped if it can't load)."""
import pytest

pytest.importorskip("sentence_transformers")

from types import SimpleNamespace

from app.config import settings
from app.services import rerank as rr


def mk(text, score=0.0):
    return SimpleNamespace(chunk=SimpleNamespace(text=text), score=score)


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    rr.reset_for_tests()
    yield
    rr.reset_for_tests()


def test_reranker_puts_the_answering_passage_first_and_scores_it_highest():
    if rr._load() is None:
        pytest.skip("cross-encoder model unavailable")
    cands = [mk("Public bicycle sharing under the Mo Cycle programme was launched."),
             mk("The Odyssey card is a common payment card for the Mo Bus service."),
             mk("Free Wi-Fi hotspots were installed in market areas.")]
    out = rr.rerank("What card do people use to pay for buses?", cands)
    assert "Odyssey" in out[0].chunk.text
    assert out[0].rerank_score > out[-1].rerank_score
    assert all(hasattr(r, "rerank_score") for r in out)


def test_reranker_fails_open_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(rr, "_load", lambda: None)
    cands = [mk("a"), mk("b")]
    assert rr.rerank("q", cands) == cands


def test_reranker_fails_open_when_scoring_raises(monkeypatch):
    class Boom:
        def predict(self, pairs):
            raise RuntimeError("OOM")
    monkeypatch.setattr(rr, "_load", lambda: Boom())
    cands = [mk("a"), mk("b")]
    assert rr.rerank("q", cands) == cands


def test_disabled_reranker_is_a_noop(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    cands = [mk("a"), mk("b")]
    assert rr.rerank("q", cands) is cands

"""Evaluation framework: metric maths, dataset integrity, reproducible run."""
import json
import subprocess
import sys
from pathlib import Path

from eval import metrics as M

ROOT = Path(__file__).resolve().parent.parent


def test_precision_recall_mrr_by_hand():
    texts = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    flags = M.relevant_flags(texts, ["gamma delta"])
    assert flags == [False, True, False, False]
    assert M.precision_at_k(flags, 2) == 0.5 and M.precision_at_k(flags, 1) == 0.0
    assert M.reciprocal_rank(flags) == 0.5
    assert M.reciprocal_rank([False, False]) == 0.0
    # multi-evidence: recall counts each evidence item
    assert M.recall_at_k(texts, ["gamma delta", "eta theta"], 2) == 0.5
    assert M.recall_at_k(texts, ["gamma delta", "eta theta"], 4) == 1.0
    assert M.recall_at_k(texts, [], 4) == 0.0


def test_abstention_metrics():
    rows = [{"answerable": True, "abstained": False}, {"answerable": True, "abstained": True},
            {"answerable": False, "abstained": True}, {"answerable": False, "abstained": False}]
    m = M.abstention_accuracy(rows)
    assert m == {"accuracy": 0.5, "false_abstain_rate": 0.5, "false_answer_rate": 0.5, "n": 4}


def test_faithfulness_detects_unsupported_sentence():
    src = ["The Odyssey card is a common payment card for the Mo Bus service."]
    ok = "The Odyssey card is a payment card for Mo Bus. [x.pdf, page 1]"
    bad = ok + " It was invented by astronauts in 1969 and costs fifty dollars."
    assert M.faithfulness(ok, src) == 1.0
    assert M.faithfulness(bad, src) == 0.5
    assert M.faithfulness("", src) == 0.0


def test_dataset_is_well_formed():
    rows = [json.loads(l) for l in (ROOT / "eval/dataset.jsonl").read_text().splitlines()]
    assert len(rows) >= 40 and len({r["id"] for r in rows}) == len(rows)
    types = {r["type"] for r in rows}
    assert {"single", "multi", "paraphrase", "unanswerable_near", "unanswerable_far"} <= types
    for r in rows:
        assert r["split"] in ("dev", "test")
        assert bool(r["evidence"]) == r["answerable"]
        if r["type"] == "multi":
            assert len({e["doc"] for e in r["evidence"]}) >= 2     # really multi-document
    assert {r["split"] for r in rows} == {"dev", "test"}


def test_eval_run_is_reproducible_and_produces_report(tmp_path):
    def run():
        subprocess.run([sys.executable, "-m", "eval.run", "--backend", "hash", "--out", str(tmp_path)],
                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
        return json.loads((tmp_path / "report-hash.json").read_text())
    a = run()
    for key in ("meta", "overall", "dev", "test", "by_type", "threshold_sweep", "failures", "questions"):
        assert key in a
    assert a["overall"]["questions"] == 56
    assert a["overall"]["citation_validity"] == 1.0   # every shown citation verifies
    b = run()
    # deterministic: same dataset and backend give identical metrics
    assert a["overall"] == b["overall"] and a["dev"] == b["dev"] and a["test"] == b["test"]

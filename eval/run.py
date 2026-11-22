"""Reproducible RAG evaluation.

    python -m eval.run --backend hash          # dependency-free embeddings
    python -m eval.run --backend st            # sentence-transformers MiniLM
    python -m eval.run --backend st --publish  # also copy the report to app/static/eval/ for the dashboard

Ingests the seed documents through the real pipeline (extract, chunk, embed,
store) into a throwaway database, then runs every question in
eval/dataset.jsonl. Offline answer mode only: no network, no API key.
Writes eval/results/report-<backend>.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hash", "st"], default="hash")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true", help="enable the cross-encoder reranker + its abstention gate")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    return ap.parse_args()


def setup_env(backend: str, rerank: bool = False) -> None:
    tmp = tempfile.mkdtemp(prefix="nexus-eval-")
    os.environ.update({
        "DATABASE_URL": "sqlite:///" + os.path.join(tmp, "eval.db"),
        "DATA_DIR": os.path.join(tmp, "data"), "STORAGE_BACKEND": "local",
        "EMBEDDING_BACKEND": backend, "LLM_MODE": "offline",
        "RERANK_ENABLED": "true" if rerank else "false", "RATE_LIMIT_ENABLED": "false",
    })


def build_corpus():
    from app.db import SessionLocal, init_db
    from app.models import Document, IngestionJob, Workspace
    from app.services import worker
    from app.services.storage import get_storage, new_key
    init_db()
    db = SessionLocal()
    ws = Workspace(name="eval", owner_id=None, is_public=1)
    db.add(ws)
    db.flush()
    for p in sorted((ROOT / "seed_docs").iterdir()):
        if p.name not in ("retrieval-augmented-generation.pdf", "odisha-handlooms.txt",
                          "bhubaneswar-smart-city.docx"):
            continue
        data = p.read_bytes()
        key = new_key(ws.id)
        get_storage().put(key, data)
        d = Document(workspace_id=ws.id, filename=p.name, storage_key=key,
                     content_hash=hashlib.sha256(data).hexdigest(), status="pending")
        db.add(d)
        db.flush()
        db.add(IngestionJob(document_id=d.id))
    db.commit()
    db.close()
    while (status := worker.run_job()) is not None:
        assert status == "done", f"ingestion failed: {status}"
    return ws.id


def load_dataset():
    return [json.loads(l) for l in (ROOT / "eval" / "dataset.jsonl").read_text().splitlines() if l.strip()]


def main() -> None:
    args = parse()
    setup_env(args.backend, args.rerank)
    sys.path.insert(0, str(ROOT))
    from eval import metrics as M
    from app.db import SessionLocal
    from app.models import Chunk
    from app.services import qa
    from app.services.embeddings import backend_name, embed, tokenize
    from app.services.retrieval import bm25_rank, vector_rank
    from app.services.citations import verify_citations
    from app.config import settings

    ws_id = build_corpus()
    db = SessionLocal()
    chunks = db.query(Chunk).filter_by(workspace_id=ws_id).order_by(Chunk.id).all()
    texts = [c.text for c in chunks]
    embs = [c.embedding_list() for c in chunks]
    data = load_dataset()

    # every gold phrase must exist in the corpus, or the label is wrong
    corpus = " ".join(texts)
    for q in data:
        for e in q["evidence"]:
            assert M.contains(corpus, e["phrase"]), f"{q['id']}: phrase not in corpus: {e['phrase']}"

    ks = [1, 3, args.k] if args.k not in (1, 3) else [1, 3]
    per_q = []
    for q in data:
        phrases = [e["phrase"] for e in q["evidence"]]
        rec = {"id": q["id"], "type": q["type"], "split": q["split"],
               "question": q["question"], "answerable": q["answerable"]}
        # --- retrieval, three systems on the same chunks
        if q["answerable"]:
            qv = embed(q["question"])
            systems = {
                "bm25": [i for i, _ in bm25_rank(texts, q["question"])],
                "vector": [i for i, _ in vector_rank(embs, qv)],
                "hybrid_rrf": [r.chunk.id for r in qa.retrieve(db, ws_id, q["question"], 20)],
            }
            by_id = {c.id: c.text for c in chunks}
            ret = {}
            for name, order in systems.items():
                ordered = [by_id[i] for i in order] if name == "hybrid_rrf" else [texts[i] for i in order]
                flags = M.relevant_flags(ordered, phrases)
                ret[name] = {"mrr": M.reciprocal_rank(flags),
                             **{f"p@{k}": M.precision_at_k(flags, k) for k in ks},
                             **{f"r@{k}": M.recall_at_k(ordered, phrases, k) for k in ks}}
            rec["retrieval"] = ret
        # --- end to end (what the API returns)
        top = qa.retrieve(db, ws_id, q["question"], args.k)
        rec["top_strength"] = round(max((r.strength for r in top), default=0.0), 4)
        rr = [r.rerank_score for r in top if hasattr(r, "rerank_score")]
        rec["top_rerank_score"] = round(max(rr), 3) if rr else None
        rec["lexical_overlap"] = bool(top and any(
            set(tokenize(q["question"])) & set(tokenize(r.chunk.text)) for r in top[:3]))
        out = qa.ask(db, ws_id, q["question"], args.k)
        rec["abstained"] = out["abstained"]
        rec["answer"] = out["answer"]
        if not out["abstained"]:
            cits = out["citations"]
            valid, problems = verify_citations(db, ws_id, cits)
            cited_texts = [c.text for c in chunks if c.id in {x.chunk_id for x in cits}]
            rec["n_citations"] = len(cits)
            rec["citations_valid"] = len(valid) == len(cits) and not problems
            if q["answerable"]:
                sup = [M.relevant_flags([t], phrases)[0] for t in cited_texts]
                rec["citation_precision"] = sum(sup) / len(sup) if sup else 0.0
                rec["citation_evidence_coverage"] = M.recall_at_k(cited_texts, phrases, len(cited_texts))
                rec["answer_evidence_coverage"] = M.answer_evidence_coverage(out["answer"], phrases)
            rec["faithfulness"] = M.faithfulness(out["answer"], cited_texts)
        per_q.append(rec)

    def summarize(rows):
        ans = [r for r in rows if r["answerable"]]
        shown = [r for r in ans if not r["abstained"]]
        allshown = [r for r in rows if not r["abstained"]]
        res = {"questions": len(rows), "answerable": len(ans),
               "unanswerable": len(rows) - len(ans),
               "retrieval": {}, "abstention": M.abstention_accuracy(rows),
               "citation_validity": M.mean(1.0 if r["citations_valid"] else 0.0 for r in allshown),
               "citation_precision": M.mean(r["citation_precision"] for r in shown),
               "citation_evidence_coverage": M.mean(r["citation_evidence_coverage"] for r in shown),
               "answer_evidence_coverage": M.mean(r["answer_evidence_coverage"] for r in shown),
               "faithfulness_heuristic": M.mean(r["faithfulness"] for r in allshown)}
        for system in ("bm25", "vector", "hybrid_rrf"):
            vals = [r["retrieval"][system] for r in ans]
            res["retrieval"][system] = {m: M.mean(v[m] for v in vals) for m in vals[0]} if vals else {}
        return res

    by_type = defaultdict(list)
    for r in per_q:
        by_type[r["type"]].append(r)
    sweep = []
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        row = {"threshold": th}
        for split in ("dev", "test"):
            rs = [{"answerable": r["answerable"], "abstained": r["top_strength"] < th}
                  for r in per_q if r["split"] == split]
            row[split] = M.abstention_accuracy(rs)["accuracy"]
        sweep.append(row)

    rerank_sweep = []
    if args.rerank:
        for th in [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 1, 2]:
            row = {"threshold": th}
            for split in ("dev", "test"):
                rs = [{"answerable": r["answerable"], "abstained": r["top_rerank_score"] < th}
                      for r in per_q if r["split"] == split]
                row[split] = M.abstention_accuracy(rs)["accuracy"]
            rerank_sweep.append(row)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    active_th = settings.abstain_threshold if backend_name().startswith("hashing") else settings.abstain_threshold_st
    report = {
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": commit, "embedding_backend": backend_name(),
                 "answer_mode": "offline (extractive, no LLM)", "k": args.k,
                 "reranker": settings.rerank_model if args.rerank else "off",
                 "abstain_rerank_threshold": settings.abstain_rerank_threshold if args.rerank else None,
                 "chunks": len(chunks), "abstain_threshold": active_th,
                 "dataset_sha256": hashlib.sha256((ROOT / "eval" / "dataset.jsonl").read_bytes()).hexdigest()[:16],
                 "notes": ["Dataset is small and written by the developer after reading the three seed documents.",
                           "faithfulness_heuristic is lexical support of answer sentences in cited chunks; "
                           "offline answers are extractive so it is near 1.0 by construction.",
                           "Claude answer mode was not evaluated (no API key)."]},
        "overall": summarize(per_q),
        "dev": summarize([r for r in per_q if r["split"] == "dev"]),
        "test": summarize([r for r in per_q if r["split"] == "test"]),
        "by_type": {t: summarize(rs) for t, rs in by_type.items()},
        "threshold_sweep": sweep,
        "rerank_threshold_sweep": rerank_sweep,
        "failures": [{k: r[k] for k in ("id", "type", "question", "abstained", "top_strength", "answer")}
                     for r in per_q if r["abstained"] == r["answerable"]],
        "questions": per_q,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report-{args.backend}{'-rerank' if args.rerank else ''}.json"
    path.write_text(json.dumps(report, indent=2))
    if args.publish:
        pub = ROOT / "app" / "static" / "eval"
        pub.mkdir(parents=True, exist_ok=True)
        (pub / path.name).write_text(json.dumps(report))
    o = report["overall"]
    print(f"backend={backend_name()} chunks={len(chunks)} wrote {path}")
    print("hybrid  :", {k: o['retrieval']['hybrid_rrf'][k] for k in ('mrr', f'p@{args.k}', f'r@{args.k}')})
    print("bm25    :", {k: o['retrieval']['bm25'][k] for k in ('mrr', f'p@{args.k}', f'r@{args.k}')})
    print("vector  :", {k: o['retrieval']['vector'][k] for k in ('mrr', f'p@{args.k}', f'r@{args.k}')})
    print("abstain :", o["abstention"], "dev", report["dev"]["abstention"]["accuracy"], "test", report["test"]["abstention"]["accuracy"])
    print("citation:", o["citation_validity"], o["citation_precision"], "faith", o["faithfulness_heuristic"], "ans_cov", o["answer_evidence_coverage"])
    print("failures:", [(f["id"], f["type"]) for f in report["failures"]])


if __name__ == "__main__":
    main()

"""Does a torch-free ONNX MiniLM give the same embeddings as sentence-transformers?

Compares sentence-transformers/all-MiniLM-L6-v2 (torch) with its published onnx/model.onnx run through
onnxruntime + tokenizers, on every eval corpus chunk and eval question. Prints cosine agreement and
whether top-5 retrieval order is identical.  pip install onnxruntime tokenizers huggingface_hub
"""
import json
import pathlib

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "sentence-transformers/all-MiniLM-L6-v2"


def texts():
    docs = [p.read_text() for p in sorted((ROOT / "seed_docs").glob("*.txt"))]
    docs += [p.read_text() for p in sorted((ROOT / "eval" / "corpus").glob("*.txt"))]
    chunks = [para.strip() for d in docs for para in d.split("\n\n") if len(para.strip()) > 40]
    qs = [json.loads(line)["question"] for line in (ROOT / "eval" / "dataset.jsonl").read_text().splitlines() if line.strip()]
    return chunks, qs


def onnx_embed(sess, tok, batch):
    tok.enable_truncation(256)
    tok.enable_padding()
    enc = tok.encode_batch(batch)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    out = sess.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
    m = mask[..., None].astype(np.float32)
    v = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


chunks, qs = texts()
st = SentenceTransformer(REPO)
st.max_seq_length = 256
sess = ort.InferenceSession(hf_hub_download(REPO, "onnx/model.onnx"), providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
a = st.encode(chunks + qs, normalize_embeddings=True)
b = onnx_embed(sess, tok, chunks + qs)
cos = (a * b).sum(1)
n = len(chunks)
same = 0
for i in range(len(qs)):
    ra = np.argsort(-(a[n + i] @ a[:n].T))[:5]
    rb = np.argsort(-(b[n + i] @ b[:n].T))[:5]
    same += int(list(ra) == list(rb))
print(f"texts={len(a)} chunks={n} questions={len(qs)}")
print(f"cosine(ST, ONNX): min={cos.min():.6f} mean={cos.mean():.6f}")
print(f"max abs diff of any component: {np.abs(a - b).max():.2e}")
print(f"top-5 order identical for {same}/{len(qs)} questions")

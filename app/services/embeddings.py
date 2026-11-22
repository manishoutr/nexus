"""Text embeddings with two interchangeable backends.

1. sentence-transformers (default when installed): all-MiniLM-L6-v2 produces
   normalized 384-dimensional semantic embeddings. The model is loaded lazily
   on first use so the API never pays the load cost at import time.
2. hashing (fallback): deterministic sparse bag-of-words vectors, no network
   and no dependencies. Keeps the demo runnable where torch is too heavy
   (e.g. a 512 MB free host). Lexical only, no real semantics.

Every chunk records which backend embedded it (Chunk.embedding_model), so a
model change is detectable and stale vectors can be re-embedded instead of
silently compared against an incompatible space.
"""
import hashlib
import logging
import math
import re
import threading
from typing import List

log = logging.getLogger("nexus.embeddings")

HASH_DIM = 256
HASH_BACKEND_NAME = "hashing-v1"
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

STOPWORDS = {"a", "an", "the", "in", "on", "at", "of", "to", "for",
             "and", "or", "with", "using", "is", "are", "by", "from",
             "as", "our", "you", "your", "we", "will", "this", "that",
             "it", "its", "be", "used", "use",
             "what", "which", "who", "whom", "whose", "when", "where",
             "why", "how", "do", "does", "did", "can", "could", "should",
             "would", "i", "me", "my", "he", "she", "they", "them", "his",
             "her", "their", "there", "here"}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed, naive plural folding."""
    tokens = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            tokens.append(tok[:-1])
    return tokens


def _embed_hash(text: str) -> List[float]:
    vec = [0.0] * HASH_DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % HASH_DIM
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class HashingBackend:
    name = HASH_BACKEND_NAME
    dim = HASH_DIM

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [_embed_hash(t) for t in texts]


class SentenceTransformerBackend:
    """all-MiniLM-L6-v2 (configurable): real semantic embeddings, normalized
    so cosine similarity is a dot product."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)
        return [v.tolist() for v in vectors]


class OnnxBackend:
    """The same all-MiniLM-L6-v2 weights run through onnxruntime, with no torch.

    Uses the model's published onnx/model.onnx plus tokenizers, mean pooling over the
    attention mask and L2 normalization, which is what sentence-transformers does for
    this model. scripts/onnx_parity.py measured cosine 1.000000 against the torch path
    on the eval texts, so vectors from either backend can sit in one index (the model
    name recorded on each chunk is therefore the same)."""

    MAX_LEN = 256

    def __init__(self, model_name: str):
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self._np = np
        self._sess = ort.InferenceSession(hf_hub_download(model_name, "onnx/model.onnx"),
                                          providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
        self._tok.enable_truncation(self.MAX_LEN)
        self._tok.enable_padding()
        self.name = model_name
        self.dim = 384
        self._run_lock = threading.Lock()  # the tokenizer's padding state is shared

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        np = self._np
        out_all = []
        for i in range(0, len(texts), 32):
            with self._run_lock:
                enc = self._tok.encode_batch(texts[i:i + 32])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            hidden = self._sess.run(None, {"input_ids": ids, "attention_mask": mask,
                                           "token_type_ids": np.zeros_like(ids)})[0]
            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            out_all.extend(x.tolist() for x in v)
        return out_all


_lock = threading.Lock()
_backend = None


def get_backend():
    """Resolve the configured embedding backend once per process.

    EMBEDDING_BACKEND=auto (default): sentence-transformers when it imports
    and the model loads, otherwise the hashing fallback.
    EMBEDDING_BACKEND=onnx: the same model through onnxruntime (no torch), fail loudly.
    EMBEDDING_BACKEND=st: require sentence-transformers (fail loudly).
    EMBEDDING_BACKEND=hash: force the fallback (used by the light demo image).
    """
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        from ..config import settings
        choice = settings.embedding_backend
        if choice == "onnx":
            _backend = OnnxBackend(settings.embedding_model)  # fail loudly if unavailable
            log.info("embedding backend: onnxruntime %s (%d-dim)", _backend.name, _backend.dim)
            return _backend
        if choice in ("auto", "st"):
            try:
                _backend = SentenceTransformerBackend(settings.embedding_model)
                log.info("embedding backend: sentence-transformers %s (%d-dim)",
                         _backend.name, _backend.dim)
                return _backend
            except Exception as e:  # noqa: BLE001 - missing dep, no network, OOM
                if choice == "st":
                    raise
                log.warning("sentence-transformers unavailable (%s)", e)
            if choice == "auto":
                try:
                    _backend = OnnxBackend(settings.embedding_model)
                    log.info("embedding backend: onnxruntime %s", _backend.name)
                    return _backend
                except Exception as e:  # noqa: BLE001
                    log.warning("onnx backend unavailable (%s); falling back to hashing embeddings", e)
        _backend = HashingBackend()
        log.info("embedding backend: %s (%d-dim)", _backend.name, _backend.dim)
        return _backend


def reset_backend_for_tests() -> None:
    global _backend
    _backend = None


def backend_name() -> str:
    return get_backend().name


def embed(text: str) -> List[float]:
    return get_backend().embed_many([text])[0]


def embed_many(texts: List[str]) -> List[List[float]]:
    return get_backend().embed_many(texts)

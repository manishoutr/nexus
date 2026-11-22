"""Central configuration. Everything environment-dependent lives here."""
import os


class Settings:
    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./nexus.db")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or None
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.llm_mode = os.getenv("LLM_MODE", "auto")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        self.max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "10"))
        # Embedding backend: auto (sentence-transformers if usable, else
        # hashing), st (require sentence-transformers) or hash (force the
        # dependency-free fallback, used by the light demo image).
        self.embedding_backend = os.getenv("EMBEDDING_BACKEND", "auto")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # Reciprocal Rank Fusion constant (standard default).
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        # Default top-k for search/ask when the caller does not pass k.
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))
        # Optional cross-encoder reranking stage (off by default: it downloads
        # a second model). Reorders the top RERANK_TOP_N candidates.
        self.rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        self.rerank_model = os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "20"))
        # Abstention thresholds: below this evidence-strength score we say
        # "not enough evidence" instead of answering. The hashing fallback is
        # lexical noise at low scores, so its threshold is lower but requires
        # a shared content word (see qa.ask). The sentence-transformers value
        # is calibrated on the seed set in tests/test_retrieval.py.
        self.abstain_threshold = float(os.getenv("ABSTAIN_THRESHOLD", "0.15"))
        self.abstain_threshold_st = float(os.getenv("ABSTAIN_THRESHOLD_ST", "0.30"))

    @property
    def claude_enabled(self) -> bool:
        if self.llm_mode == "offline":
            return False
        if self.llm_mode == "claude" and not self.anthropic_api_key:
            raise RuntimeError("LLM_MODE=claude but ANTHROPIC_API_KEY is not set")
        return bool(self.anthropic_api_key)


settings = Settings()

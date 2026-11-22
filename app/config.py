"""Central configuration. Everything environment-dependent lives here."""
import os


class Settings:
    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./nexus.db")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or None
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.llm_mode = os.getenv("LLM_MODE", "auto")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        # Storage: local (DATA_DIR) or s3 (any S3-compatible endpoint).
        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        self.s3_bucket = os.getenv("S3_BUCKET", "nexus-documents")
        self.s3_endpoint_url = os.getenv("S3_ENDPOINT_URL") or None
        self.s3_region = os.getenv("AWS_REGION", "us-east-1")
        # Job queue: db (threaded poller, no extra services) or celery (Redis).
        self.queue_backend = os.getenv("QUEUE_BACKEND", "db")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.job_max_attempts = int(os.getenv("JOB_MAX_ATTEMPTS", "3"))
        self.job_backoff_base_s = float(os.getenv("JOB_BACKOFF_BASE_S", "2"))
        # A job stuck in "processing" longer than this is presumed orphaned
        # by a crashed worker and goes back to the queue.
        self.job_stale_after_s = float(os.getenv("JOB_STALE_AFTER_S", "300"))
        # Auth. JWT_SECRET must be set in production; without it a random
        # per-process secret is generated (tokens die on restart).
        self.jwt_secret = os.getenv("JWT_SECRET") or None
        self.access_token_minutes = int(os.getenv("ACCESS_TOKEN_MINUTES", "30"))
        self.refresh_token_days = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
        # Rate limits (slowapi syntax). Storage can be redis://... so the
        # limits hold across several API processes; memory:// is per-process.
        self.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.rate_limit_storage = os.getenv("RATE_LIMIT_STORAGE", "memory://")
        self.rate_limit_auth = os.getenv("RATE_LIMIT_AUTH", "10/minute")
        self.rate_limit_ask = os.getenv("RATE_LIMIT_ASK", "20/minute")
        self.rate_limit_upload = os.getenv("RATE_LIMIT_UPLOAD", "30/minute")
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

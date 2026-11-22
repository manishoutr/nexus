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
        # Retrieval abstention threshold: below this combined score we say
        # "not enough evidence" instead of answering. Tuned on the seed set.
        self.abstain_threshold = float(os.getenv("ABSTAIN_THRESHOLD", "0.15"))

    @property
    def claude_enabled(self) -> bool:
        if self.llm_mode == "offline":
            return False
        if self.llm_mode == "claude" and not self.anthropic_api_key:
            raise RuntimeError("LLM_MODE=claude but ANTHROPIC_API_KEY is not set")
        return bool(self.anthropic_api_key)


settings = Settings()

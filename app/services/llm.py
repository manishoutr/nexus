"""LLM service layer: the only file that talks to the Anthropic SDK.

Model output is validated against a pydantic schema before use; on any
failure callers fall back to deterministic logic. Without an API key the app
runs in OFFLINE mode and answers are labelled as such.
"""
import json
import logging
import time
from typing import List, Optional, Type

from pydantic import BaseModel, ValidationError

from ..config import settings

log = logging.getLogger("nexus.llm")

try:
    import anthropic
except ImportError:
    anthropic = None

TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
RETRY_BASE_S = 1.0


class LLMError(Exception):
    """A classified LLM failure. `kind` is one of: timeout, rate_limited,
    auth, api_error, invalid_output. Callers fall back to offline answers."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


def classify(exc: Exception) -> str:
    name = type(exc).__name__
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if "RateLimit" in name:
        return "rate_limited"
    if "Authentication" in name or "PermissionDenied" in name:
        return "auth"
    return "api_error"


_RETRYABLE = {"timeout", "rate_limited", "api_error"}


class LLMService:
    def __init__(self, client=None, enabled: Optional[bool] = None) -> None:
        # `client` / `enabled` can be injected so tests drive the failure paths
        # without a network or an API key.
        if client is not None:
            self.enabled = True if enabled is None else enabled
            self._client = client
        else:
            self.enabled = settings.claude_enabled and anthropic is not None
            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key, timeout=TIMEOUT_SECONDS
            ) if self.enabled else None
        self.mode = "claude" if self.enabled else "offline"

    def complete_json(self, system: str, user: str,
                      schema: Type[BaseModel]) -> BaseModel:
        """Ask Claude for JSON validated against `schema`.

        Raises LLMError (never returns None, never hangs past the SDK timeout):
        timeouts, rate limits and 5xx are retried with backoff; auth errors
        and invalid output are not.
        """
        if not self.enabled:
            raise LLMError("disabled", "no API key configured")
        prompt = user + "\n\nRespond with ONLY a JSON object, no markdown fences."
        last: Optional[LLMError] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.messages.create(
                    model=settings.anthropic_model, max_tokens=2048, system=system,
                    messages=[{"role": "user", "content": prompt}])
            except Exception as e:  # noqa: BLE001 - classify every SDK failure
                last = LLMError(classify(e), str(e)[:200])
                log.warning("claude call failed (attempt %d): %s", attempt + 1, last)
                if last.kind in _RETRYABLE and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_S * 2 ** attempt)
                    continue
                raise last
            try:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return schema.model_validate(json.loads(_strip_fences(text)))
            except (ValidationError, json.JSONDecodeError, AttributeError) as e:
                raise LLMError("invalid_output", str(e)[:200])
        raise last or LLMError("api_error")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


llm = LLMService()

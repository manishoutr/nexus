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


class LLMService:
    def __init__(self) -> None:
        self.enabled = settings.claude_enabled and anthropic is not None
        self.mode = "claude" if self.enabled else "offline"
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=TIMEOUT_SECONDS
        ) if self.enabled else None

    def complete_json(self, system: str, user: str,
                      schema: Type[BaseModel]) -> Optional[BaseModel]:
        """Ask Claude for JSON, validate against `schema`, else None."""
        if not self.enabled:
            return None
        prompt = user + "\n\nRespond with ONLY a JSON object, no markdown fences."
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                return schema.model_validate(json.loads(text))
            except (ValidationError, json.JSONDecodeError) as e:
                log.warning("claude output failed validation: %s", e)
                return None
            except Exception as e:  # noqa: BLE001 - rate limits, network, 5xx
                log.warning("claude call failed (attempt %d): %s", attempt, e)
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
        return None


llm = LLMService()

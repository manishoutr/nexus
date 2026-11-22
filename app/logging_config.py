"""Structured logging: one JSON object per line (LOG_FORMAT=json, the default
in production) or plain text for local development."""
import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

request_id_var = contextvars.ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        out = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name}
        try:  # worker code already logs JSON strings; merge instead of nesting
            parsed = json.loads(msg)
            out.update(parsed if isinstance(parsed, dict) else {"message": msg})
        except (ValueError, TypeError):
            out["message"] = msg
        rid = request_id_var.get()
        if rid:
            out["request_id"] = rid
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "json").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

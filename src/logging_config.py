"""Structured logging configuration.

Provides a JSON-aware console formatter plus optional file output so that
logs from the training pipeline, API, and dashboard are consistent and
machine-parseable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .config import settings


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with a stable timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


class ContextFilter(logging.Filter):
    """Attach context (e.g. run id, stage) to every record."""

    context: dict[str, Any] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


context_filter = ContextFilter()


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure root logging once with console + optional file handlers."""
    log_level = (level or settings.log_level).upper()
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(log_level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    console.addFilter(context_filter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger bound to the ``house_price`` namespace."""
    return logging.getLogger(f"house_price.{name}")


def set_log_context(**kwargs: Any) -> None:
    """Add key/value context to all subsequently emitted log records."""
    context_filter.context.update(kwargs)

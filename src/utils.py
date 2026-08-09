"""General-purpose helpers shared across the platform."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .logging_config import get_logger

logger = get_logger(__name__)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)


@dataclass
class Timer:
    """Simple wall-clock timer that logs when used as a context manager."""

    name: str = "block"
    _start: float = field(default=0.0, init=False)

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info("%s completed in %.2fs", self.name, elapsed)


@contextmanager
def timed(name: str = "block") -> Iterator[Timer]:
    timer = Timer(name)
    with timer:
        yield timer


def as_dataframe(data: Any) -> pd.DataFrame:
    """Coerce common input shapes into a DataFrame with row ids."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, (list, tuple)):
        return pd.DataFrame(data)
    raise TypeError(f"Cannot convert {type(data)} to DataFrame")


def save_plot(fig: Any, path: str | Path, dpi: int = 120) -> Path:
    """Save a matplotlib figure to disk and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def ensure_serialisable(value: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ensure_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serialisable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"

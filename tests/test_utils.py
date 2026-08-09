"""Tests for shared utilities."""

from __future__ import annotations

import numpy as np
from src.utils import ensure_serialisable, load_json, save_json


def test_json_roundtrip(tmp_path):
    payload = {"a": 1, "b": [1, 2], "c": {"d": np.float64(2.5)}}
    path = tmp_path / "x.json"
    save_json(payload, path)
    loaded = load_json(path)
    assert loaded == {"a": 1, "b": [1, 2], "c": {"d": 2.5}}


def test_ensure_serialisable_converts_numpy():
    out = ensure_serialisable({"x": np.int64(5), "y": np.float32(1.5)})
    assert out == {"x": 5, "y": 1.5}

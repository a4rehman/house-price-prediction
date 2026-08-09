"""Tests for the MLflow registry and local artifact persistence."""

from __future__ import annotations

import numpy as np
from src.data.preprocessing import Preprocessor
from src.models.registry import (
    list_registered_versions,
    load_local_artifacts,
    log_training_run,
    save_local_artifacts,
)
from src.models.training import build_models


def test_local_artifact_roundtrip(ames_synthetic, tmp_path):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model = build_models(random_state=0)["Ridge"]
    model.fit(Xt, np.log1p(y))

    directory = save_local_artifacts(
        model, pre, {"model_name": "Ridge_test", "score": 1.0}, tmp_path,
    )
    artifacts = load_local_artifacts(directory)
    assert artifacts["metadata"]["model_name"] == "Ridge_test"

    preds = artifacts["preprocessor"].transform(X.iloc[:5])
    assert artifacts["model"].predict(preds).shape == (5,)


def test_log_training_run_mlflow(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model = build_models(random_state=0)["Ridge"]
    model.fit(Xt, np.log1p(y))

    run_info = log_training_run(
        experiment_name="house_price_prediction",
        run_name="test_run",
        params={"alpha": 1.0},
        metrics={"rmse": 1000.0},
        tags={"env": "test"},
        model=model,
        preprocessor=pre,
        register=True,
    )
    assert run_info["run_id"]
    assert run_info["model_uri"]

    versions = list_registered_versions()
    assert any(v["stage"] == "None" or v["version"] for v in versions)

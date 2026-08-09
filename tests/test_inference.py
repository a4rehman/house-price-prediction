"""Tests for the prediction service (inference)."""

from __future__ import annotations

import pandas as pd
from src.models.inference import PredictionService
from src.models.registry import load_local_artifacts


def test_service_loads_and_predicts(tiny_artifacts):
    service = PredictionService(local_dir=tiny_artifacts).load()
    assert service.is_loaded()

    result = service.predict({"OverallQual": 6, "GrLivArea": 1500, "LotArea": 8000})
    assert isinstance(result, pd.DataFrame)
    assert "predicted_price" in result.columns
    assert result["predicted_price"].iloc[0] > 0


def test_service_batch_predict(tiny_artifacts, ames_synthetic):
    service = PredictionService(local_dir=tiny_artifacts).load()
    sample = ames_synthetic.head(20).drop(columns=["SalePrice"])
    out = service.predict_batch(sample)
    assert len(out) == 20
    assert (out["predicted_price"] > 0).all()


def test_service_single_with_explanation(tiny_artifacts):
    service = PredictionService(local_dir=tiny_artifacts).load()
    result = service.predict_single(
        {"OverallQual": 8, "GrLivArea": 2200, "Neighborhood": "NridgHt"}
    )
    assert result["predicted_price"] > 0
    assert "explanation" in result


def test_service_handles_missing_features(tiny_artifacts):
    service = PredictionService(local_dir=tiny_artifacts).load()
    result = service.predict({"OverallQual": 7})
    assert result["predicted_price"].iloc[0] > 0


def test_local_artifacts_roundtrip(tiny_artifacts):
    artifacts = load_local_artifacts(tiny_artifacts)
    assert "model" in artifacts
    assert "preprocessor" in artifacts
    assert artifacts["metadata"]["model_name"] == "RandomForest_test"

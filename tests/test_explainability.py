"""Tests for SHAP explainability."""

from __future__ import annotations

import numpy as np
from src.data.preprocessing import Preprocessor
from src.models.explainability import global_importance, local_explanation
from src.models.training import build_models


def test_global_importance(ames_synthetic, tmp_path):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model = build_models(random_state=0)["RandomForest"]
    model.set_params(n_estimators=10, max_depth=5)
    model.fit(Xt, np.log1p(y))

    report = global_importance(model, Xt, plots_dir=tmp_path, sample_size=50)
    assert "feature_importance" in report
    assert len(report["feature_importance"]) > 0
    assert (tmp_path / "shap_summary.png").exists()
    assert (tmp_path / "shap_importance.png").exists()


def test_local_explanation(ames_synthetic, tmp_path):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model = build_models(random_state=0)["RandomForest"]
    model.set_params(n_estimators=10, max_depth=5)
    model.fit(Xt, np.log1p(y))

    sample = X.iloc[[0]]
    explanation = local_explanation(model, pre, sample)
    assert "base_value" in explanation
    assert "explanation" in explanation
    assert len(explanation["explanation"]) > 0

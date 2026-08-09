"""Tests for Optuna-based hyperparameter tuning."""

from __future__ import annotations

from src.data.preprocessing import Preprocessor
from src.models.tuning import tune_model


def test_tune_model_returns_best_params(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    best_params, result = tune_model(
        "RandomForest", Xt, y, n_trials=2, show_progress=False,
    )
    assert isinstance(best_params, dict)
    assert "best_cv_rmse_log" in result
    assert result["best_cv_rmse_log"] > 0


def test_tune_model_unknown_model(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)
    try:
        tune_model("NotAModel", Xt, y, n_trials=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

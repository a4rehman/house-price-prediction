"""Tests for the modelling layer."""

from __future__ import annotations

import numpy as np
from src.data.preprocessing import Preprocessor
from src.models.comparison import compare_models
from src.models.metrics import inverse_log_target, log_target, regression_metrics
from src.models.training import build_models, cross_validate, train_final_model


def test_log_roundtrip():
    prices = np.array([100.0, 25_000.0, 400_000.0])
    np.testing.assert_allclose(inverse_log_target(log_target(prices)), prices, rtol=1e-9)


def test_regression_metrics_perfect():
    y_true = np.array([100_000, 200_000, 300_000], dtype=float)
    metrics = regression_metrics(y_true, y_true)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_build_models_contains_core_families():
    models = build_models(random_state=0)
    assert {"Ridge", "ElasticNet", "RandomForest"} <= set(models)


def test_cross_validate_returns_summary(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)
    model = build_models(random_state=0)["Ridge"]
    result = cross_validate(model, Xt, np.log1p(y), cv=2)
    assert set(result) == {"cv_rmse_log_mean", "cv_rmse_log_std", "cv_folds"}
    assert result["cv_folds"] == 2
    assert result["cv_rmse_log_mean"] > 0


def test_train_final_model_evaluates_validation(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model, metrics = train_final_model(
        build_models(random_state=0)["Ridge"],
        Xt.iloc[:200], y.iloc[:200],
        Xt.iloc[200:], y.iloc[200:],
    )
    assert {"rmse", "mae", "r2", "rmse_log"} <= set(metrics)


def test_compare_models_produces_leaderboard(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)
    lb = compare_models(Xt, y, verbose=False)
    assert "model" in lb.columns
    assert len(lb) >= 3
    assert lb["cv_rmse_log_mean"].is_monotonic_increasing

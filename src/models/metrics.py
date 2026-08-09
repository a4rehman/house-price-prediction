"""Regression metrics used to evaluate and compare models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def log_target(y: np.ndarray) -> np.ndarray:
    """Apply the log1p transform used for the target variable."""
    return np.log1p(y)


def inverse_log_target(y_log: np.ndarray) -> np.ndarray:
    """Invert the log1p transform back to the original price scale."""
    return np.expm1(y_log)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R2 (and % error) on the original price scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 3),
    }


def regression_metrics_log(
    y_true: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, float]:
    """Report metrics where the model output is on the log-price scale."""
    y_true = np.asarray(y_true, dtype=float)
    rmse_log = float(np.sqrt(mean_squared_error(log_target(y_true), y_pred_log)))
    return {
        "rmse_log": round(rmse_log, 5),
        "r2_log": round(float(r2_score(log_target(y_true), y_pred_log)), 4),
    }


def merged_metrics(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Combined metrics dict for a model that predicts in log-space."""
    y_pred = inverse_log_target(y_pred_log)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(regression_metrics_log(y_true, y_pred_log))
    return metrics

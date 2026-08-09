"""Model comparison: cross-validated leaderboard + diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import settings
from ..logging_config import get_logger
from ..utils import save_plot
from .metrics import inverse_log_target
from .training import build_models, cross_validate

logger = get_logger(__name__)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    plots_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every candidate with k-fold CV and return a leaderboard."""
    plots_dir = Path(plots_dir or settings.plots_dir)
    models = build_models()
    rows: list[dict] = []

    for name, model in models.items():
        try:
            result = cross_validate(model, X, y)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Model %s failed: %s", name, exc)
            continue
        rows.append({"model": name, **result})
        if verbose:
            logger.info(
                "%-18s CV RMSE(log) = %.4f (± %.4f)",
                name, result["cv_rmse_log_mean"], result["cv_rmse_log_std"],
            )

    leaderboard = pd.DataFrame(rows).sort_values("cv_rmse_log_mean").reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(leaderboard))))
    order = leaderboard.sort_values("cv_rmse_log_mean")["model"]
    sns.barplot(
        data=leaderboard, x="cv_rmse_log_mean", y="model", order=order,
        hue="model", legend=False, palette="viridis", ax=ax,
    )
    ax.set_title("Model comparison — cross-validated RMSE (log scale)")
    ax.set_xlabel("CV RMSE (log1p price)")
    fig.tight_layout()
    save_plot(fig, plots_dir / "model_comparison.png")

    csv_path = plots_dir.parent / "reports" / "model_leaderboard.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(csv_path, index=False)
    logger.info("Leaderboard saved to %s", csv_path)
    return leaderboard


def evaluate_holdout(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_log_target: bool = True,
    plots_dir: Path | None = None,
) -> dict:
    """Score a fitted model on a held-out set and save diagnostic plots."""
    from .metrics import merged_metrics

    plots_dir = Path(plots_dir or settings.plots_dir)
    preds_log = model.predict(X_test)
    if not use_log_target:
        preds_log = np.log1p(np.clip(preds_log, 0, None))
    y_pred = inverse_log_target(preds_log)
    y_true = pd.Series(y_test, index=X_test.index)

    metrics = merged_metrics(y_true.to_numpy(), preds_log)
    logger.info("Hold-out metrics: %s", metrics)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=15, color="#2c3e50")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1.5)
    axes[0].set_xlabel("Actual price")
    axes[0].set_ylabel("Predicted price")
    axes[0].set_title("Predicted vs actual")
    axes[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    resid = y_true - y_pred
    axes[1].scatter(y_pred, resid, alpha=0.4, s=15, color="#16a085")
    axes[1].axhline(0, color="r", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Predicted price")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals")
    axes[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_plot(fig, plots_dir / "holdout_diagnostics.png")

    return metrics

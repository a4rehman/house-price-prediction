"""Model explainability with SHAP.

Produces global (summary/bar) and local (waterfall/force) explanations that
are reused by the CLI, the REST API, and the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import settings
from ..logging_config import get_logger
from ..utils import ensure_serialisable, save_plot

logger = get_logger(__name__)


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def build_explainer(model: Any, X_background: pd.DataFrame) -> Any:
    """Create a SHAP explainer suited to the model family."""
    if not _shap_available():
        raise RuntimeError("SHAP is not installed — `pip install shap`.")
    import shap

    model_type = type(model).__name__.lower()
    if any(k in model_type for k in ("xgb", "lgb", "catboost", "rf", "gb", "randomforest", "gradientboosting")):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            return shap.Explainer(model, X_background)
    return shap.Explainer(model, X_background)


def global_importance(
    model: Any,
    X: pd.DataFrame,
    plots_dir: str | Path | None = None,
    sample_size: int = 1000,
) -> dict[str, Any]:
    """Compute mean absolute SHAP values and save summary + bar plots."""
    plots_dir = Path(plots_dir or settings.plots_dir)
    import shap

    X_sampled = X.sample(n=min(sample_size, len(X)), random_state=settings.random_state)
    explainer = build_explainer(model, X_sampled)
    shap_values = explainer(X_sampled)

    importance = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=shap_values.feature_names,
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sampled, show=False, max_display=20)
    save_plot(fig, plots_dir / "shap_summary.png")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.bar(shap_values, max_display=20, show=False)
    save_plot(fig, plots_dir / "shap_importance.png")
    plt.close("all")

    logger.info("SHAP global importance computed for %d features", len(importance))
    return {
        "feature_importance": ensure_serialisable(importance.head(30).to_dict()),
        "base_value": float(np.mean(shap_values.base_values)),
    }


def local_explanation(
    model: Any,
    preprocessor: Any,
    X_sample: pd.DataFrame,
    raw_sample: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Explain a single prediction with a SHAP waterfall summary."""
    import shap

    X_transformed = preprocessor.transform(X_sample)
    explainer = build_explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    values = np.asarray(shap_values.values[0])
    base = float(np.asarray(shap_values.base_values)[0])
    names = [str(n) for n in shap_values.feature_names]

    top_idx = np.argsort(np.abs(values))[::-1][:15]
    top = [
        {"feature": names[i], "value": float(values[i])}
        for i in top_idx
        if not np.isnan(values[i])
    ]

    try:
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        save_plot(fig, settings.plots_dir / "shap_waterfall.png")
    except Exception as exc:  # pragma: no cover
        logger.warning("Waterfall plot failed: %s", exc)
    finally:
        plt.close("all")

    return {
        "base_value": base,
        "explanation": top,
        "expected_value": base,
        "feature_names": names,
    }

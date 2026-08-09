"""Model definitions, cross-validation, and final-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from ..config import settings
from ..logging_config import get_logger
from .metrics import log_target

logger = get_logger(__name__)

try:  # Optional-but-recommended gradient boosting backends
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


def build_models(random_state: int | None = None) -> dict[str, Any]:
    """Return a dictionary of {name: unfitted model} candidates."""
    seed = random_state or settings.random_state
    models: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "Lasso": Lasso(alpha=0.001, random_state=seed),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "SVR": SVR(C=100.0, gamma="scale"),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, verbosity=0,
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_state=seed, verbose=False, allow_writing_files=False,
        )
    return models


def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int | None = None,
    scoring: str = "neg_root_mean_squared_error",
) -> dict[str, float]:
    """Run k-fold CV on the (log-transformed) target and return the summary.

    The target is transformed to log-space first (when ``use_log_target`` is
    enabled) so that every caller reports a consistent log-scale RMSE.
    """
    n_folds = cv or settings.cv_folds
    y_score = log_target(y) if settings.use_log_target else y
    scores = cross_val_score(
        model, X, y_score, cv=KFold(n_splits=n_folds, shuffle=True,
                                    random_state=settings.random_state),
        scoring=scoring, n_jobs=-1,
    )
    return {
        "cv_rmse_log_mean": float(-scores.mean()),
        "cv_rmse_log_std": float(scores.std()),
        "cv_folds": int(n_folds),
    }


def train_final_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    use_log_target: bool | None = None,
) -> tuple[Any, dict[str, float]]:
    """Fit a model on all training rows; optionally evaluate on validation."""
    use_log = settings.use_log_target if use_log_target is None else use_log_target
    y_target = log_target(y_train) if use_log else y_train
    model.fit(X_train, y_target)

    if X_val is not None and y_val is not None:
        from .metrics import merged_metrics

        preds_log = model.predict(X_val)
        if not use_log:
            preds_log = np.log1p(np.maximum(preds_log, 0))
        metrics = merged_metrics(np.asarray(y_val), np.asarray(preds_log))
        logger.info("Validation metrics: %s", metrics)
        return model, metrics

    logger.info("Model trained on %d samples", len(X_train))
    return model, {}


def save_model_artifacts(
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
    model_name: str = "best_model",
    directory: str | Path | None = None,
) -> Path:
    """Persist model + preprocessor + metadata to the artifacts directory."""
    directory = Path(directory or settings.models_dir / model_name)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    metadata["model_name"] = model_name
    metadata["use_log_target"] = settings.use_log_target

    import json

    with open(directory / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    logger.info("Model artifacts saved to %s", directory)
    return directory

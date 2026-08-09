"""Hyperparameter optimisation with Optuna."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from ..config import settings
from ..logging_config import get_logger
from .metrics import log_target

logger = get_logger(__name__)


def _make_objective(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    base_model: Any,
) -> Callable[[Any], float]:
    """Build an Optuna objective that tunes a given model family."""

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        if model_name in {"XGBoost", "LightGBM", "GradientBoosting"}:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        if model_name in {"XGBoost", "LightGBM", "RandomForest", "GradientBoosting"}:
            params["min_child_samples" if model_name != "RandomForest" else "min_samples_leaf"] = (
                trial.suggest_int("min_samples", 2, 30)
            )
        if model_name in {"XGBoost", "LightGBM"}:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        if model_name == "LightGBM":
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 128)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        if model_name == "XGBoost":
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        if model_name == "RandomForest":
            params["max_features"] = trial.suggest_float("max_features", 0.3, 1.0)
            params["max_depth"] = trial.suggest_int("max_depth", 5, 40)
        if model_name == "CatBoost":
            params["depth"] = trial.suggest_int("depth", 4, 10)
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            params["iterations"] = trial.suggest_int("iterations", 100, 800, step=50)
        if model_name == "SVR":
            params["C"] = trial.suggest_float("C", 1e-2, 1e3, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        if model_name in {"Ridge", "Lasso", "ElasticNet"}:
            params["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            if model_name == "ElasticNet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.01, 0.99)

        kwargs: dict[str, Any] = {}
        if model_name in {"RandomForest", "GradientBoosting", "Ridge", "Lasso", "ElasticNet"}:
            kwargs["random_state"] = settings.random_state
        if model_name in {"XGBoost", "LightGBM", "RandomForest"}:
            kwargs["n_jobs"] = -1
        if model_name in {"XGBoost", "LightGBM"}:
            kwargs["verbose"] = 0
        if model_name == "CatBoost":
            kwargs = {"random_seed": settings.random_state, "verbose": False,
                      "allow_writing_files": False}

        model = base_model.__class__(**params, **kwargs)

        y_score = log_target(y) if settings.use_log_target else y
        kf = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
        scores = cross_val_score(
            model, X, y_score, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        return float(-scores.mean())

    return objective


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    timeout_seconds: int | None = None,
    show_progress: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Tune a model family with Optuna; returns (best_model, best_params)."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Optuna is required for tuning — `pip install optuna`."
        ) from exc

    from .training import build_models

    base_model = build_models().get(model_name)
    if base_model is None:
        raise ValueError(f"Unknown model '{model_name}'")

    n_trials = n_trials or settings.n_trials

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_state),
        study_name=f"tune_{model_name}",
    )
    study.optimize(
        _make_objective(model_name, X, y, base_model),
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=show_progress,
    )

    logger.info(
        "Tuned %s in %d trials: best CV RMSE(log)=%.4f",
        model_name, len(study.trials), study.best_value,
    )
    return study.best_params, {"best_cv_rmse_log": float(study.best_value)}

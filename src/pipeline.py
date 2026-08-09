"""End-to-end training pipeline orchestrator.

Loads data, runs EDA, cleans outliers, compares models, tunes the winner,
trains the final model, evaluates on a hold-out set, computes SHAP values,
and registers the artifact both locally and in MLflow.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings
from .data.eda import run_eda
from .data.loader import download_raw_data, load_raw_data
from .data.preprocessing import Preprocessor, remove_outliers
from .logging_config import get_logger, set_log_context
from .models.comparison import compare_models, evaluate_holdout
from .models.explainability import global_importance
from .models.metrics import log_target
from .models.registry import log_training_run, save_local_artifacts
from .models.tuning import tune_model

logger = get_logger(__name__)


def load_and_split(
    force_download: bool = False,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any]:
    """Load raw data, clean outliers, and split into train/test."""
    download_raw_data(force=force_download)
    raw = load_raw_data()

    cleaned = remove_outliers(raw, target_col=settings.target_column)

    report = run_eda(cleaned)
    logger.info(
        "Dataset after cleaning: %d rows, %d columns",
        cleaned.shape[0], cleaned.shape[1],
    )

    y = cleaned[settings.target_column]
    X = cleaned.drop(columns=[settings.target_column])

    split = train_test_split(
        X, y, test_size=test_size or settings.test_size,
        random_state=settings.random_state,
    )
    X_train, X_test, y_train, y_test = split

    # Save processed copies for reproducibility / batch jobs.
    cleaned.to_csv(settings.data_processed_dir / "dataset.csv", index=False)
    X_train.to_csv(settings.data_processed_dir / "train.csv", index=False)
    X_test.to_csv(settings.data_processed_dir / "test.csv", index=False)
    y_train.to_csv(settings.data_processed_dir / "train_target.csv", index=False)
    y_test.to_csv(settings.data_processed_dir / "test_target.csv", index=False)

    logger.info(
        "Train/Test split: %d / %d rows", len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test, report


def run_training(
    tune: bool = True,
    n_trials: int | None = None,
    force_download: bool = False,
    register_mlflow: bool = True,
    skip_eda: bool = False,
) -> dict[str, Any]:
    """Execute the full training pipeline and return a results summary."""
    set_log_context(stage="pipeline", run="main")
    logger.info("=== Starting House Price training pipeline ===")

    X_train, X_test, y_train, y_test, _ = load_and_split(force_download=force_download)

    preprocessor = Preprocessor(scale=True).fit(X_train, y_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # ---- 1. Baseline comparison ------------------------------------------
    logger.info("--- Model comparison (k-fold CV) ---")
    leaderboard = compare_models(X_train_t, y_train, verbose=True)
    best_base = leaderboard.iloc[0]["model"]
    best_base_cv = float(leaderboard.iloc[0]["cv_rmse_log_mean"])
    logger.info("Best baseline model: %s (CV RMSE log=%.4f)", best_base, best_base_cv)

    # ---- 2. Hyperparameter tuning ------------------------------------------
    final_params: dict[str, Any] = {}
    tuned_cv: dict[str, float] = {}
    if tune:
        logger.info("--- Tuning %s with Optuna (%d trials) ---", best_base, n_trials)
        final_params, tuned_cv = tune_model(
            best_base, X_train_t, y_train, n_trials=n_trials,
        )
        logger.info("Best params for %s: %s", best_base, final_params)
    else:
        tuned_cv = {"best_cv_rmse_log": best_base_cv}

    # ---- 3. Train final model ----------------------------------------------
    logger.info("--- Training final model ---")
    from .models.training import build_models

    base_model = build_models()[best_base]
    kwargs = {"random_state": settings.random_state}
    if best_base in {"XGBoost", "LightGBM", "RandomForest"}:
        kwargs["n_jobs"] = -1
    if best_base in {"XGBoost", "LightGBM"}:
        kwargs["verbose"] = 0
    if best_base == "CatBoost":
        kwargs = {"random_seed": settings.random_state, "verbose": False,
                  "allow_writing_files": False}
    final_model = base_model.__class__(**final_params, **kwargs)

    y_train_log = log_target(y_train)
    final_model.fit(X_train_t, y_train_log)

    # ---- 4. Hold-out evaluation ---------------------------------------------
    logger.info("--- Hold-out evaluation ---")
    holdout_metrics = evaluate_holdout(
        final_model, X_test_t, y_test,
        use_log_target=True, plots_dir=settings.plots_dir,
    )

    # ---- 5. SHAP explainability ----------------------------------------------
    logger.info("--- SHAP global explainability ---")
    shap_report = {}
    try:
        shap_report = global_importance(
            final_model, X_train_t, plots_dir=settings.plots_dir,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP analysis skipped: %s", exc)

    # ---- 6. Persist locally --------------------------------------------------
    metadata = {
        "model_name": f"{best_base}_tuned",
        "best_params": final_params,
        "cv_rmse_log": tuned_cv.get("best_cv_rmse_log"),
        "holdout": holdout_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": settings.random_state,
        "shap": shap_report,
    }
    artifacts_dir = save_local_artifacts(
        final_model, preprocessor, metadata,
        directory=settings.models_dir / "best_model",
    )

    # ---- 7. Register in MLflow ------------------------------------------------
    mlflow_run: dict[str, Any] = {}
    if register_mlflow:
        logger.info("--- Logging run to MLflow ---")
        mlflow_run = log_training_run(
            experiment_name="house_price_prediction",
            run_name=f"{best_base}_tuned",
            params={"best_model": best_base, **final_params},
            metrics={**holdout_metrics, **tuned_cv},
            tags={"stage": "production", "framework": best_base},
            model=final_model,
            preprocessor=preprocessor,
            artifacts={"plots": str(settings.plots_dir), "reports": str(settings.reports_dir)},
            register=True,
        )

    # ---- 8. Save leaderboard + summary ----------------------------------------
    leaderboard.to_csv(settings.reports_dir / "model_leaderboard.csv", index=False)
    summary = {
        "best_model": best_base,
        "best_params": final_params,
        "cv_rmse_log": tuned_cv.get("best_cv_rmse_log"),
        "holdout_metrics": holdout_metrics,
        "artifacts_dir": str(artifacts_dir),
        "mlflow": mlflow_run,
        "shap_importance": shap_report.get("feature_importance", {}),
    }
    with open(settings.reports_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=== Pipeline complete. Best model: %s ===", best_base)
    return summary

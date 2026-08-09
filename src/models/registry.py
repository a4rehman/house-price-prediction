"""Model registry: MLflow experiment tracking + artifact persistence.

The registry logs every training run to MLflow, registers the best model
under a stable name, and can re-load any registered version at serving time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def _mlflow():
    try:
        import mlflow
        return mlflow
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("MLflow is not installed — `pip install mlflow`.") from exc


def set_tracking_uri(uri: str | None = None) -> str:
    mlflow = _mlflow()
    # MLflow >= 3.x requires an explicit opt-in for the local file store.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    # Default to a SQLite backend so both tracking AND the model registry work
    # on Windows and Linux alike.
    sqlite_uri = f"sqlite:///{settings.mlruns_dir.resolve().as_posix()}/mlflow.db"
    resolved = uri or sqlite_uri
    mlflow.set_tracking_uri(resolved)
    mlflow.set_experiment("house_price_prediction")
    logger.info("MLflow tracking URI: %s", resolved)
    return resolved


def log_training_run(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    model=None,
    preprocessor=None,
    artifacts: dict[str, str] | None = None,
    register: bool = True,
    log_model_signature: Any = None,
) -> dict[str, Any]:
    """Start an MLflow run, log everything, and optionally register a model."""
    mlflow = _mlflow()
    set_tracking_uri()
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        for key, value in (params or {}).items():
            mlflow.log_param(key, value)
        for key, value in (metrics or {}).items():
            mlflow.log_metric(key, value)
        for name, path in (artifacts or {}).items():
            mlflow.log_artifact(path, artifact_path=name)

        model_uri = None
        if model is not None and preprocessor is not None:
            model_uri = _log_model(
                mlflow, model, preprocessor, register,
                signature=log_model_signature,
            )

        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": model_uri,
            "params": params,
            "metrics": metrics,
        }
        logger.info("MLflow run %s logged", run.info.run_id)
        return run_info


def _log_model(mlflow, model: Any, preprocessor: Any, register: bool,
               signature: Any = None) -> str:
    """Log the model (with the preprocessor as its pre-transform) to MLflow."""

    class _Wrapper(mlflow.pyfunc.PythonModel):
        """PyFunc wrapper so predictions always take raw DataFrames."""

        def __init__(self, model, preprocessor, use_log_target: bool):
            self.model = model
            self.preprocessor = preprocessor
            self.use_log_target = use_log_target

        def predict(self, context, model_input: Any) -> Any:
            import numpy as np
            import pandas as pd

            df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
            X = self.preprocessor.transform(df)
            preds = self.model.predict(X)
            if self.use_log_target:
                preds = np.expm1(preds)
            return preds

    wrapper = _Wrapper(model, preprocessor, settings.use_log_target)
    pyfunc_params = {
        "name": "model",
        "python_model": wrapper,
        "code_paths": [str(settings.project_root / "src")],
        "registered_model_name": settings.registered_model_name if register else None,
    }
    if signature is not None:
        pyfunc_params["signature"] = signature
    mlflow.pyfunc.log_model(**pyfunc_params)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    logger.info("Model logged to %s", model_uri)
    return model_uri


def load_registered_model(version: str = "latest", name: str | None = None) -> Any:
    """Load a registered model as a pyfunc wrapper (raw DF in, prices out)."""
    mlflow = _mlflow()
    set_tracking_uri()
    name = name or settings.registered_model_name
    model_uri = f"models:/{name}/{version}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def list_registered_versions(name: str | None = None) -> list[dict[str, Any]]:
    """List every registered version of the house price model."""
    mlflow = _mlflow()
    set_tracking_uri()
    name = name or settings.registered_model_name
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:  # pragma: no cover - model not yet registered
        return []
    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
            "created": str(v.creation_timestamp),
        }
        for v in versions
    ]


# ---------------------------------------------------------------------------
# Local artifact helpers (used when MLflow is not required)
# ---------------------------------------------------------------------------

def save_local_artifacts(
    model: Any,
    preprocessor: Any,
    metadata: dict[str, Any],
    directory: str | Path | None = None,
) -> Path:
    """Persist model + preprocessor + metadata to a local directory."""
    directory = Path(directory or settings.models_dir / "best_model")
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    with open(directory / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Local artifacts saved to %s", directory)
    return directory


def load_local_artifacts(directory: str | Path | None = None) -> dict[str, Any]:
    """Load locally persisted model, preprocessor, and metadata."""
    directory = Path(directory or settings.models_dir / "best_model")
    model_path = directory / "model.joblib"
    preprocessor_path = directory / "preprocessor.joblib"
    if not (model_path.exists() and preprocessor_path.exists()):
        raise FileNotFoundError(
            f"No trained artifacts found at {directory}. "
            "Run `python scripts/run_training.py` first."
        )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata: dict[str, Any] = {}
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"model": model, "preprocessor": preprocessor, "metadata": metadata}

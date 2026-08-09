"""Prediction service shared by the REST API and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import settings
from ..logging_config import get_logger
from ..utils import as_dataframe
from .explainability import local_explanation
from .metrics import inverse_log_target
from .registry import load_local_artifacts, load_registered_model

logger = get_logger(__name__)


class PredictionService:
    """Encapsulates model loading, transformation, prediction, and explanation.

    Modes:
      * ``local``  — loads ``artifacts/models/best_model`` (fast, no MLflow).
      * ``mlflow`` — loads a registered version by URI (defaults to latest).
    """

    def __init__(
        self,
        source: str = "local",
        model_uri: str = "",
        local_dir: str | Path | None = None,
    ) -> None:
        self.source = source
        self.model_uri = model_uri or settings.model_uri
        self.local_dir = Path(local_dir or settings.models_dir / "best_model")
        self.model = None
        self.preprocessor = None
        self.metadata: dict[str, Any] = {}
        self._use_log_target = settings.use_log_target
        self._loaded = False

    # -- loading -----------------------------------------------------------
    def load(self) -> PredictionService:
        if self._loaded:
            return self

        if self.source == "mlflow" or self.model_uri.startswith("mlflow://"):
            uri = self.model_uri.replace("mlflow://", "")
            wrapper = load_registered_model(uri or "latest")
            self.model = wrapper
            self.preprocessor = None
            self.metadata = {"source": "mlflow", "model_uri": uri}
            logger.info("Loaded MLflow model '%s'", uri)
        else:
            artifacts = load_local_artifacts(self.local_dir)
            self.model = artifacts["model"]
            self.preprocessor = artifacts["preprocessor"]
            self.metadata = artifacts.get("metadata", {})
            logger.info("Loaded local model from %s", self.local_dir)

        self._loaded = True
        return self

    # -- prediction ---------------------------------------------------------
    def predict(self, data: pd.DataFrame | dict | list) -> pd.DataFrame:
        """Return a DataFrame with raw input id, prediction, and confidence."""
        self.load()
        df = as_dataframe(data)
        raw_ids = df[settings.id_column] if settings.id_column in df.columns else None

        if self.preprocessor is not None:
            X = self.preprocessor.transform(df)
            preds_log = self.model.predict(X)
            if self._use_log_target:
                preds_log = np.maximum(preds_log, 0)
            prices = inverse_log_target(preds_log)
        else:  # MLflow pyfunc wrapper expects raw DataFrame
            prices = np.asarray(self.model.predict(df), dtype=float).ravel()

        result = pd.DataFrame({"predicted_price": prices.round(2)})
        if raw_ids is not None:
            result.insert(0, settings.id_column, raw_ids.values)
        else:
            result.insert(0, settings.id_column, np.arange(1, len(result) + 1))
        return result

    def predict_single(self, data: dict) -> dict[str, Any]:
        """Predict one house and include a SHAP explanation."""
        df = as_dataframe(data)
        prediction = self.predict(df).iloc[0]

        explanation = {}
        if self.preprocessor is not None:
            try:
                explanation = local_explanation(self.model, self.preprocessor, df)
            except Exception as exc:  # pragma: no cover
                logger.warning("SHAP explanation failed: %s", exc)
                explanation = {"error": str(exc)}

        return {
            "predicted_price": float(prediction["predicted_price"]),
            "prediction": float(prediction["predicted_price"]),
            "explanation": explanation,
            "model": self.metadata.get("model_name", "best_model"),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict multiple rows at once (returns a DataFrame)."""
        return self.predict(df)

    def is_loaded(self) -> bool:
        return self._loaded

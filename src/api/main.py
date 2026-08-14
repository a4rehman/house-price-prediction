"""FastAPI application entry point.

Exposes a REST API for single, batch, and CSV predictions plus SHAP
explanations and model registry introspection. The model is loaded lazily on
first request so the API boots even before a model has been trained.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..logging_config import get_logger, setup_logging
from ..models.inference import PredictionService
from .endpoints import health, models, predict

logger = get_logger(__name__)
setup_logging(log_file=None)


@dataclass
class AppState:
    """Shared, process-wide state."""

    service: PredictionService = None  # type: ignore[assignment]


app_state = AppState()


def create_app() -> FastAPI:
    app = FastAPI(
        title="House Price Prediction API",
        description=(
            "Production-ready REST API for the Ames house price prediction "
            "platform. Supports single predictions, batch predictions, CSV "
            "uploads, SHAP explanations, and model registry introspection."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_state.service = PredictionService()

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(models.router)

    @app.on_event("startup")
    async def _startup() -> None:
        # Warm the model so first request is fast; failures are tolerated.
        try:
            app_state.service.load()
            logger.info("Model loaded at startup: %s",
                        app_state.service.metadata.get("model_name"))
        except Exception as exc:
            logger.warning("Model not loaded at startup: %s", exc)

    return app


app = create_app()

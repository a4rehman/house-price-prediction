"""Health and root endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ... import __version__
from ...models.inference import PredictionService
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])

_start_time = time.time()


def get_service() -> PredictionService:
    from ..main import app_state

    return app_state.service


@router.get("/", summary="Root")
def root() -> dict:
    return {
        "service": "House Price Prediction API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/predict/csv",
            "/api/v1/explain",
            "/api/v1/models",
        ],
    }


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(service: PredictionService = Depends(get_service)) -> HealthResponse:
    try:
        service.load()
        loaded = service.is_loaded()
        source = service.metadata.get("source", "local")
    except Exception:
        loaded, source = False, "unavailable"

    return HealthResponse(
        status="ok" if loaded else "degraded",
        service="house-price-api",
        version=__version__,
        model_loaded=loaded,
        model_source=source,
    )

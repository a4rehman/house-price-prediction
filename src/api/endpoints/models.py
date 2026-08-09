"""Model registry listing endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from ...config import settings
from ...logging_config import get_logger
from ...models.inference import PredictionService
from ...models.registry import list_registered_versions
from ..schemas import ModelInfo

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


def get_service() -> PredictionService:
    from ..main import app_state

    return app_state.service


@router.get("", summary="List available models and registered versions")
def models(service: PredictionService = Depends(get_service)) -> dict:
    try:
        service.load()
        local_meta = service.metadata
    except Exception:
        local_meta = {"error": "no local model trained"}

    versions = list_registered_versions()

    registered = [
        ModelInfo(
            name=settings.registered_model_name,
            version=str(v["version"]),
            stage=v.get("stage"),
            registered_at=v.get("created"),
        )
        for v in versions
    ]

    # Local metadata may contain numpy/json types; keep it serialisable.
    try:
        meta = json.loads(json.dumps(local_meta, default=str))
    except Exception:
        meta = {"note": "metadata unavailable"}

    return {
        "active": {
            "model": local_meta.get("model_name", "unknown"),
            "source": service.metadata.get("source", "local"),
        },
        "registered": [r.model_dump() for r in registered],
        "local_metadata": meta,
    }

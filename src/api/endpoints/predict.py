"""Prediction endpoints: single, batch, CSV upload, and explanation."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...config import settings
from ...logging_config import get_logger
from ...models.inference import PredictionService
from ..schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HouseFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])


def get_service() -> PredictionService:
    from ..main import app_state

    return app_state.service


def _predict_frame(service: PredictionService, df: pd.DataFrame) -> list[BatchPredictionItem]:
    result = service.predict_batch(df)
    return [
        BatchPredictionItem(
            id=int(row[settings.id_column]),
            predicted_price=float(row["predicted_price"]),
        )
        for _, row in result.iterrows()
    ]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict price for a single house",
)
def predict(
    features: HouseFeatures,
    service: PredictionService = Depends(get_service),
) -> PredictionResponse:
    try:
        df = features.to_dataframe()
        out = service.predict_single(df)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    explanation = None
    if out.get("explanation") and "error" not in out["explanation"]:
        from ..schemas import Explanation, ShapFeature

        explanation = Explanation(
            base_value=out["explanation"]["base_value"],
            expected_value=out["explanation"]["expected_value"],
            explanation=[
                ShapFeature(feature=item["feature"], value=item["value"])
                for item in out["explanation"]["explanation"]
            ],
        )

    return PredictionResponse(
        id=int(features.Id) if features.Id is not None else 1,
        predicted_price=out["predicted_price"],
        prediction=out["prediction"],
        model=out["model"],
        explanation=explanation,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict prices for multiple houses",
)
def predict_batch(
    features: list[HouseFeatures],
    service: PredictionService = Depends(get_service),
) -> BatchPredictionResponse:
    try:
        df = pd.DataFrame([f.to_dataframe().iloc[0] for f in features])
        items = _predict_frame(service, df)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return BatchPredictionResponse(count=len(items), predictions=items)


@router.post(
    "/predict/csv",
    response_model=BatchPredictionResponse,
    summary="Upload a CSV of houses and get predictions",
)
async def predict_csv(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_service),
) -> BatchPredictionResponse:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    if df.shape[0] > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Too many rows ({df.shape[0]}); max is {settings.max_batch_size}",
        )

    try:
        items = _predict_frame(service, df)
    except Exception as exc:
        logger.exception("CSV batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("CSV '%s' predicted %d rows", file.filename, len(items))
    return BatchPredictionResponse(count=len(items), predictions=items)


@router.post("/explain", summary="SHAP explanation for a single house")
def explain(
    features: HouseFeatures,
    service: PredictionService = Depends(get_service),
) -> dict:
    try:
        df = features.to_dataframe()
        out = service.predict_single(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "predicted_price": out["predicted_price"],
        "explanation": out.get("explanation", {}),
    }

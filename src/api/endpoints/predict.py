"""Authenticated prediction and administration endpoints."""
from __future__ import annotations
import io
import os
import pandas as pd
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from ...churn import ChurnService, prepare
from ..schemas import CustomerFeatures, PredictionResponse

router = APIRouter(prefix="/api/v1", tags=["churn prediction"])

def service() -> ChurnService:
    from ..main import app_state
    assert app_state.service is not None
    return app_state.service

def authenticate(x_api_key: str | None = Header(default=None)) -> None:
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post("/predict", response_model=PredictionResponse, dependencies=[Depends(authenticate)])
def predict(customer: CustomerFeatures, model: ChurnService = Depends(service)) -> dict:
    return model.predict(customer.model_dump())

@router.post("/predict/batch", dependencies=[Depends(authenticate)])
def predict_batch(customers: list[CustomerFeatures], model: ChurnService = Depends(service)) -> dict:
    return {"count": len(customers), "predictions": [model.predict(x.model_dump()) for x in customers]}

@router.post("/predict/csv", dependencies=[Depends(authenticate)])
async def predict_csv(file: UploadFile = File(...), model: ChurnService = Depends(service)) -> dict:
    df = prepare(pd.read_csv(io.BytesIO(await file.read())))
    return {"count": len(df), "predictions": [model.predict(row.dropna().to_dict()) for _, row in df.iterrows()]}

@router.post("/admin/train", dependencies=[Depends(authenticate)], tags=["admin"])
async def train(file: UploadFile | None = File(default=None), model: ChurnService = Depends(service)) -> dict:
    frame = pd.read_csv(io.BytesIO(await file.read())) if file else None
    return model.train(frame)

@router.get("/admin/metrics", dependencies=[Depends(authenticate)], tags=["admin"])
def metrics(model: ChurnService = Depends(service)) -> dict:
    if model.metrics is None: model.load()
    if model.metrics is None: return model.train()
    return model.metrics

@router.post("/explain", dependencies=[Depends(authenticate)])
def explain(customer: CustomerFeatures, model: ChurnService = Depends(service)) -> dict:
    result = model.predict(customer.model_dump())
    return {**result, "feature_impacts": model.explain(customer.model_dump())}

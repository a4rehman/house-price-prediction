"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Raw house attributes. Every field is optional with a neutral default.

    The fitted :class:`Preprocessor` imputes any missing value, so a minimal
    payload (e.g. just ``OverallQual``) is always accepted.
    """

    customer_id: str = "NEW-CUSTOMER"
    tenure: int = Field(default=12, ge=0, le=120)
    monthly_charges: float = Field(default=70, ge=0)
    total_charges: float = Field(default=840, ge=0)
    support_tickets: int = Field(default=0, ge=0)
    contract: str = "Month-to-month"
    internet_service: str = "DSL"
    payment_method: str = "Electronic check"
    senior_citizen: int = Field(default=0, ge=0, le=1)
    paperless_billing: str = "Yes"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class ShapFeature(BaseModel):
    feature: str
    value: float


class Explanation(BaseModel):
    base_value: float
    expected_value: float
    explanation: list[ShapFeature]


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_category: str
    recommendations: list[str]


class BatchPredictionItem(BaseModel):
    customer_id: str
    churn_probability: float
    risk_category: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_source: str


class ModelInfo(BaseModel):
    name: str
    version: str
    stage: str | None = None
    registered_at: str | None = None

"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Raw house attributes. Every field is optional with a neutral default.

    The fitted :class:`Preprocessor` imputes any missing value, so a minimal
    payload (e.g. just ``OverallQual``) is always accepted.
    """

    Id: int | None = None
    MSSubClass: float | None = 0.0
    MSZoning: str | None = None
    LotFrontage: float | None = None
    LotArea: float | None = 0.0
    Street: str | None = None
    Alley: str | None = None
    LotShape: str | None = None
    LandContour: str | None = None
    Utilities: str | None = None
    LotConfig: str | None = None
    LandSlope: str | None = None
    Neighborhood: str | None = None
    Condition1: str | None = None
    Condition2: str | None = None
    BldgType: str | None = None
    HouseStyle: str | None = None
    OverallQual: int | None = Field(default=5, ge=0, le=10)
    OverallCond: int | None = Field(default=5, ge=0, le=10)
    YearBuilt: float | None = None
    YearRemodAdd: float | None = None
    RoofStyle: str | None = None
    RoofMatl: str | None = None
    Exterior1st: str | None = None
    Exterior2nd: str | None = None
    MasVnrType: str | None = None
    MasVnrArea: float | None = 0.0
    ExterQual: str | None = None
    ExterCond: str | None = None
    Foundation: str | None = None
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: float | None = 0.0
    BsmtFinType2: str | None = None
    BsmtFinSF2: float | None = 0.0
    BsmtUnfSF: float | None = 0.0
    TotalBsmtSF: float | None = 0.0
    Heating: str | None = None
    HeatingQC: str | None = None
    CentralAir: str | None = None
    Electrical: str | None = None
    FirstFlrSF: float | None = Field(default=0.0, alias="1stFlrSF")
    SecondFlrSF: float | None = Field(default=0.0, alias="2ndFlrSF")
    LowQualFinSF: float | None = 0.0
    GrLivArea: float | None = 0.0
    BsmtFullBath: float | None = 0.0
    BsmtHalfBath: float | None = 0.0
    FullBath: float | None = 0.0
    HalfBath: float | None = 0.0
    BedroomAbvGr: float | None = 0.0
    KitchenAbvGr: float | None = 0.0
    KitchenQual: str | None = None
    TotRmsAbvGrd: float | None = 0.0
    Functional: str | None = None
    Fireplaces: float | None = 0.0
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: float | None = 0.0
    GarageArea: float | None = 0.0
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str | None = None
    WoodDeckSF: float | None = 0.0
    OpenPorchSF: float | None = 0.0
    EnclosedPorch: float | None = 0.0
    ThreeSsnPorch: float | None = Field(default=0.0, alias="3SsnPorch")
    ScreenPorch: float | None = 0.0
    PoolArea: float | None = 0.0
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: float | None = 0.0
    MoSold: float | None = None
    YrSold: float | None = None
    SaleType: str | None = None
    SaleCondition: str | None = None

    model_config = {"populate_by_name": True}

    def to_dataframe(self) -> pd.DataFrame:
        data = self.model_dump(by_alias=True, exclude_none=True)
        df = pd.DataFrame([data])
        for col in ("1stFlrSF", "2ndFlrSF", "3SsnPorch"):
            if col not in df.columns:
                df[col] = 0.0
        return df


class ShapFeature(BaseModel):
    feature: str
    value: float


class Explanation(BaseModel):
    base_value: float
    expected_value: float
    explanation: list[ShapFeature]


class PredictionResponse(BaseModel):
    id: int
    predicted_price: float
    prediction: float
    model: str
    explanation: Explanation | None = None


class BatchPredictionItem(BaseModel):
    id: int
    predicted_price: float


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

"""Domain-driven feature engineering for the Ames housing dataset.

Each engineered feature is a plain pandas expression so the same code runs
identically during training, validation, and serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numeric features produced by :func:`engineer_features`. Used by the
# preprocessor so engineered columns are also scaled and modelled.
ENGINEERED_NUMERIC = [
    "TotalSF", "TotalFinSF", "Has2ndFloor", "LotFrontagePerArea",
    "TotalBath", "TotalPorchSF", "HouseAge", "RemodAge", "Remodelled",
    "IsNew", "GarageAge", "HasPool", "HasGarage", "HasBasement",
    "HasFireplace", "OverallGrade", "QualPerSF", "BedroomRatio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a copy of ``df``."""
    out = df.copy()

    # ---- Size aggregates ---------------------------------------------------
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(out.columns):
        out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    if {"GrLivArea", "TotalBsmtSF"}.issubset(out.columns):
        out["TotalFinSF"] = out["GrLivArea"] + out["TotalBsmtSF"]
    if {"1stFlrSF", "2ndFlrSF"}.issubset(out.columns):
        out["Has2ndFloor"] = (out["2ndFlrSF"] > 0).astype(int)
    if {"LotFrontage", "LotArea"}.issubset(out.columns):
        out["LotFrontagePerArea"] = out["LotFrontage"] / (out["LotArea"] + 1)

    # ---- Bathrooms ---------------------------------------------------------
    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(out.columns):
        out["TotalBath"] = (
            out["FullBath"]
            + 0.5 * out["HalfBath"]
            + out["BsmtFullBath"]
            + 0.5 * out["BsmtHalfBath"]
        )

    # ---- Porches / outdoor living ------------------------------------------
    porch_cols = ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"]
    if all(c in out.columns for c in porch_cols):
        out["TotalPorchSF"] = out[porch_cols].sum(axis=1)

    # ---- Age & renovation ---------------------------------------------------
    if {"YrSold", "YearBuilt"}.issubset(out.columns):
        out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    if {"YrSold", "YearRemodAdd"}.issubset(out.columns):
        out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    if {"YearBuilt", "YearRemodAdd"}.issubset(out.columns):
        out["Remodelled"] = (out["YearRemodAdd"] != out["YearBuilt"]).astype(int)
    if {"YearBuilt", "YrSold"}.issubset(out.columns):
        out["IsNew"] = (out["YearBuilt"] == out["YrSold"]).astype(int)
    if {"GarageYrBlt", "YrSold"}.issubset(out.columns):
        out["GarageAge"] = (out["YrSold"] - out["GarageYrBlt"]).clip(lower=0)

    # ---- Flags -------------------------------------------------------------
    if "PoolArea" in out.columns:
        out["HasPool"] = (out["PoolArea"] > 0).astype(int)
    if "GarageArea" in out.columns:
        out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in out.columns:
        out["HasBasement"] = (out["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in out.columns:
        out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)

    # ---- Composite quality score -------------------------------------------
    if {"OverallQual", "OverallCond"}.issubset(out.columns):
        out["OverallGrade"] = out["OverallQual"] * out["OverallCond"]
    if {"OverallQual", "TotalSF"}.issubset(out.columns):
        out["QualPerSF"] = out["OverallQual"] / (out["TotalSF"] + 1)

    # ---- Bedrooms-per-room density ------------------------------------------
    if {"BedroomAbvGr", "TotRmsAbvGrd"}.issubset(out.columns):
        out["BedroomRatio"] = out["BedroomAbvGr"] / (out["TotRmsAbvGrd"] + 1)

    # Sanity: replace any resulting inf with NaN so imputation handles it.
    return out.replace([np.inf, -np.inf], np.nan)

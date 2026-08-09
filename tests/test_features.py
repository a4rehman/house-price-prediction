"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd
from src.features.engineering import engineer_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TotalBsmtSF": [1000, 0],
            "1stFlrSF": [1200, 800],
            "2ndFlrSF": [800, 0],
            "GrLivArea": [2000, 800],
            "FullBath": [2, 1],
            "HalfBath": [1, 0],
            "BsmtFullBath": [1, 0],
            "BsmtHalfBath": [0, 0],
            "YrSold": [2010, 2008],
            "YearBuilt": [2000, 1950],
            "YearRemodAdd": [2005, 1950],
            "GarageYrBlt": [2005, 0],
            "PoolArea": [0, 400],
            "GarageArea": [600, 0],
            "Fireplaces": [1, 0],
            "OverallQual": [7, 4],
            "OverallCond": [5, 6],
            "BedroomAbvGr": [3, 2],
            "TotRmsAbvGrd": [8, 5],
            "LotFrontage": [60, 50],
            "LotArea": [9000, 5000],
            "WoodDeckSF": [200, 0],
            "OpenPorchSF": [100, 0],
            "3SsnPorch": [0, 0],
            "EnclosedPorch": [0, 0],
            "ScreenPorch": [0, 0],
        }
    )


def test_total_sf_is_sum():
    out = engineer_features(_frame())
    expected = pd.Series([1000 + 1200 + 800, 0 + 800 + 0])
    pd.testing.assert_series_equal(out["TotalSF"], expected, check_names=False)


def test_total_bath():
    out = engineer_features(_frame())
    expected = pd.Series([2 + 0.5 * 1 + 1, 1 + 0.0 + 0])
    pd.testing.assert_series_equal(out["TotalBath"], expected, check_names=False)


def test_flags():
    out = engineer_features(_frame())
    assert list(out["HasPool"]) == [0, 1]
    assert list(out["HasGarage"]) == [1, 0]
    assert list(out["HasFireplace"]) == [1, 0]
    assert list(out["IsNew"]) == [0, 0]


def test_age_features():
    out = engineer_features(_frame())
    assert list(out["HouseAge"]) == [10, 58]
    assert list(out["Remodelled"]) == [1, 0]


def test_input_unchanged():
    df = _frame()
    engineer_features(df)
    assert "TotalSF" not in df.columns

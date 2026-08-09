"""Tests for the preprocessing layer."""

from __future__ import annotations

import numpy as np
from src.data.preprocessing import (
    Preprocessor,
    fill_missing_values,
    remove_outliers,
)


def test_fill_missing_values_removes_all_nan(ames_synthetic):
    df = ames_synthetic.copy()
    # Introduce NaNs in numeric and categorical columns.
    df.loc[0, "LotFrontage"] = np.nan
    df.loc[1, "KitchenQual"] = np.nan
    df.loc[2, "PoolQC"] = np.nan

    filled = fill_missing_values(df)
    assert not filled.isna().any().any()


def test_fill_missing_none_categories_become_none_string(ames_synthetic):
    df = ames_synthetic.copy()
    df["PoolQC"] = np.nan
    filled = fill_missing_values(df)
    assert set(filled["PoolQC"].unique()) == {"None"}


def test_remove_outliers_caps_lot_area(ames_synthetic):
    df = ames_synthetic.copy()
    df["LotArea"] = 1_000_000
    cleaned = remove_outliers(df, target_col="SalePrice")
    assert cleaned["LotArea"].max() <= 100_000


def test_preprocessor_fit_transform_shape(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    assert Xt.shape[1] == len(pre.feature_names_)
    assert list(Xt.columns) == pre.feature_names_
    assert not Xt.isna().any().any()


def test_preprocessor_consistency_across_splits(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    X_train = X.iloc[:200]
    X_test = X.iloc[200:]

    pre = Preprocessor(scale=True).fit(X_train, y.iloc[:200])
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)

    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert list(Xt_train.columns) == list(Xt_test.columns)


def test_preprocessor_handles_new_categories(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    pre = Preprocessor(scale=True).fit(X, y)
    weird = X.iloc[[0]].copy()
    weird["Neighborhood"] = "BrandNewTown"
    Xt = pre.transform(weird)
    assert not Xt.isna().any().any()


def test_preprocessor_rejects_transform_before_fit(ames_synthetic):
    pre = Preprocessor(scale=True)
    try:
        pre.transform(ames_synthetic)
        raise AssertionError("Expected RuntimeError before fit")
    except RuntimeError:
        pass


def test_preprocessor_feature_engineering_included(ames_synthetic):
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    pre = Preprocessor(scale=True).fit(X, y)
    assert "TotalSF" in pre.feature_names_
    assert "TotalBath" in pre.feature_names_
    assert "HasGarage" in pre.feature_names_

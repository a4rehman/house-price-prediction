"""Preprocessing: missing values, outliers, encoding, and scaling.

The :class:`Preprocessor` is fitted on the training set and then re-applied
identically to validation, test, batch, and single-record inputs so that the
exact same feature space is produced at every stage of the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ..config import settings
from ..features.engineering import ENGINEERED_NUMERIC, engineer_features
from ..logging_config import get_logger

logger = get_logger(__name__)

# Columns whose NaN literally means "this feature does not exist".
NONE_FILL_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]

# Numeric columns where NaN should be replaced by 0 ("feature absent").
NONE_FILL_NUMERIC = [
    "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea",
    "GarageYrBlt", "MasVnrArea",
]

# Explicit ordinal category rankings (highest = best / most).
ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "ExterQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "MiscFeature": ["None", "Othr", "Shed", "TenC", "Gar2", "Elev"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "PavedDrive": ["N", "P", "Y"],
    "Street": ["Grvl", "Pave"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "LandContour": ["Low", "HLS", "Bnk", "Lvl"],
    "Condition1": [
        "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA",
        "RRNe", "RRAe",
    ],
    "Condition2": [
        "Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA",
        "RRNe", "RRAe",
    ],
}

# Remaining categorical columns treated as nominal (one-hot encoded).
NOMINAL_COLUMNS = [
    "MSZoning", "MSSubClass", "Neighborhood", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "GarageType", "SaleType",
    "SaleCondition", "LotConfig", "Electrical", "Alley",
]

# Columns dropped because they are ids / useless for prediction.
DROP_COLUMNS = ["Id", "PID", "Order"]

# Always numeric columns (scaled).
NUMERIC_COLUMNS = [
    "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal", "MoSold", "YrSold", "OverallQual", "OverallCond",
]


def _is_category_with_none(col: str, values: pd.Series) -> bool:
    if col not in NONE_FILL_CATEGORICAL:
        return False
    # Infer whether values use "None"/NaN to encode absence.
    return bool(values.isna().any() or (values.astype(str) == "None").any())


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using Ames-domain-aware strategies."""
    out = df.copy()

    # 1. Categorical "does not exist" -> explicit "None" string.
    for col in NONE_FILL_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].fillna("None").astype(str)

    # 2. Numeric "does not exist" -> 0.
    for col in NONE_FILL_NUMERIC:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    # 3. LotFrontage -> median by neighborhood, else global median.
    if "LotFrontage" in out.columns:
        lot = out.groupby("Neighborhood")["LotFrontage"].transform("median")
        global_median = float(out["LotFrontage"].median())
        out["LotFrontage"] = out["LotFrontage"].fillna(lot).fillna(global_median)

    # 4. Any residual numeric NaNs -> column median.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    # 5. Any residual object NaNs -> most frequent value.
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode()[0])

    return out


def remove_outliers(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    keep_flags: bool = False,
) -> pd.DataFrame:
    """Remove/drop true data errors and cap extreme values.

    Strategy:
      * Drop the well-known GrLivArea > 4000 with SalePrice < 300k pairs.
      * Cap LotArea to a sane ceiling (known erroneous records).
      * IQR-based winsorization of long-tailed numeric features.
    """
    out = df.copy()
    target = target_col if target_col in out.columns else None

    before = len(out)

    # Known data errors (overly large, cheap houses).
    if target and "GrLivArea" in out.columns:
        mask = (out["GrLivArea"] > 4000) & (out[target] < 300_000)
        if mask.any():
            logger.info("Removing %d suspicious cheap-large houses", int(mask.sum()))
            out = out[~mask]

    if "LotArea" in out.columns:
        cap = settings.cap_extreme_lot_area
        n = int((out["LotArea"] > cap).sum())
        if n:
            logger.info("Capping %d extreme LotArea values at %d", n, cap)
            out["LotArea"] = out["LotArea"].clip(upper=cap)

    # IQR winsorization on long-tailed features (affects rows, not dropped).
    tail_cols = [
        "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF",
        "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF", "PoolArea",
    ]
    for col in tail_cols:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - settings.outlier_iqr_multiplier * iqr, q3 + settings.outlier_iqr_multiplier * iqr
        clipped = out[col].clip(lower=lo, upper=hi)
        if keep_flags:
            out[f"{col}_was_outlier"] = (out[col] != clipped).astype(int)
        out[col] = clipped

    if target:
        # Winsorize the target to stabilise training on the log scale.
        q1, q3 = out[target].quantile([0.25, 0.75])
        hi = q3 + 3.0 * (q3 - q1)
        out[target] = out[target].clip(upper=hi)

    logger.info(
        "Outlier pass complete: %d -> %d rows (%.1f%% kept)",
        before, len(out), 100 * len(out) / max(before, 1),
    )
    return out


@dataclass
class Preprocessor:
    """Fitted feature transformer shared between training and inference.

    Attributes:
        scale: Whether numeric columns are standardised.
        ordinal_encoder: fitted OrdinalEncoder for ordinal columns.
        onehot_encoder: fitted OneHotEncoder for nominal columns.
        scaler: fitted StandardScaler for numeric columns.
        numeric_columns_: numeric columns actually present.
        ordinal_columns_: ordinal columns actually present.
        nominal_columns_: nominal columns actually present.
        feature_names_: final ordered feature names.
    """

    scale: bool = True
    ordinal_encoder: Any = field(default_factory=lambda: OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1,
    ))
    onehot_encoder: Any = field(default_factory=lambda: OneHotEncoder(
        handle_unknown="ignore", drop="first", sparse_output=False,
    ))
    scaler: Any = field(default_factory=StandardScaler)
    numeric_columns_: list[str] = field(default_factory=list)
    ordinal_columns_: list[str] = field(default_factory=list)
    nominal_columns_: list[str] = field(default_factory=list)
    onehot_feature_names_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # Imputation statistics captured from the training set.
    _numeric_center_: dict[str, float] = field(default_factory=dict)
    _object_center_: dict[str, str] = field(default_factory=dict)
    _lot_frontage_map_: Any = None
    _lot_frontage_fallback_: float = 0.0

    # ---- helpers ----------------------------------------------------------
    def _available(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if c in df.columns]

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        return fill_missing_values(df)

    def _clean_absent_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the deterministic 'feature does not exist' fills."""
        for col in NONE_FILL_CATEGORICAL:
            if col in X.columns:
                X[col] = X[col].fillna("None").astype(str)
        for col in NONE_FILL_NUMERIC:
            if col in X.columns:
                X[col] = X[col].fillna(0.0).astype(float)
        return X

    def _impute_using_training_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training-time imputation values to any remaining NaN."""
        if "LotFrontage" in X.columns and self._lot_frontage_map_ is not None:
            X["LotFrontage"] = (
                X["LotFrontage"]
                .fillna(X["Neighborhood"].map(self._lot_frontage_map_))
                .fillna(self._lot_frontage_fallback_)
            )
        for col, value in self._numeric_center_.items():
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(value)
        for col, value in self._object_center_.items():
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(value)
        # Final safety: encoders must never see NaN.
        for col in X.select_dtypes(include=["object"]).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(self._object_center_.get(col, "None"))
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(self._numeric_center_.get(col, 0.0))
        return X

    def _ensure_all_known_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add any known feature column missing from the input with defaults.

        Numeric columns are added at their training median (a neutral value);
        categorical columns are added at their most common training value.
        """
        known = (
            set(self.numeric_columns_)
            | set(self.ordinal_columns_)
            | set(self.nominal_columns_)
        )
        for col in known:
            if col not in X.columns:
                if col in self.ordinal_columns_ or col in self.nominal_columns_:
                    X[col] = self._object_center_.get(col, "None")
                else:
                    X[col] = self._numeric_center_.get(col, 0.0)
        return X

    # ---- API --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Preprocessor:
        X = self._select_columns(self._fill(X))
        X = engineer_features(X)

        self.numeric_columns_ = self._available(
            X, NUMERIC_COLUMNS + ENGINEERED_NUMERIC
        )
        self.ordinal_columns_ = self._available(X, list(ORDINAL_CATEGORIES))
        self.nominal_columns_ = self._available(X, NOMINAL_COLUMNS)

        # Capture imputation statistics from the training set.
        for col in self.numeric_columns_:
            self._numeric_center_[col] = float(X[col].median())
        for col in X.select_dtypes(include=["object"]).columns:
            self._object_center_[col] = str(X[col].mode().iloc[0])
        if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            self._lot_frontage_map_ = X.groupby("Neighborhood")["LotFrontage"].median()
            self._lot_frontage_fallback_ = float(X["LotFrontage"].median())

        if self.ordinal_columns_:
            cats = [ORDINAL_CATEGORIES[c] for c in self.ordinal_columns_]
            self.ordinal_encoder.set_params(categories=cats)
            self.ordinal_encoder.fit(X[self.ordinal_columns_])

        if self.nominal_columns_:
            self.onehot_encoder.fit(X[self.nominal_columns_])
            drop_idx = self.onehot_encoder.drop_idx_
            self.onehot_feature_names_ = []
            for i, (col, cats) in enumerate(
                zip(
                    self.onehot_encoder.feature_names_in_,
                    self.onehot_encoder.categories_,
                    strict=True,
                )
            ):
                dropped = drop_idx[i] if drop_idx is not None else None
                for j, cat in enumerate(cats):
                    if dropped is not None and j == dropped:
                        continue
                    self.onehot_feature_names_.append(f"{col}_{cat}")

        if self.scale and self.numeric_columns_:
            self.scaler.fit(X[self.numeric_columns_])

        self.feature_names_ = (
            list(self.numeric_columns_)
            + list(self.ordinal_columns_)
            + self.onehot_feature_names_
        )
        self.fitted_ = True
        logger.info(
            "Preprocessor fitted: %d features (%d numeric, %d ordinal, %d one-hot)",
            len(self.feature_names_), len(self.numeric_columns_),
            len(self.ordinal_columns_), len(self.onehot_feature_names_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        X = self._select_columns(X.copy())
        X = self._clean_absent_features(X)
        X = engineer_features(X)
        X = self._ensure_all_known_columns(X)
        X = self._impute_using_training_stats(X)

        parts: list[pd.DataFrame] = []

        if self.numeric_columns_:
            num = X[self.numeric_columns_].astype(float)
            if self.scale:
                num = pd.DataFrame(
                    self.scaler.transform(num),
                    columns=self.numeric_columns_,
                    index=X.index,
                )
            parts.append(num)

        if self.ordinal_columns_:
            ord_df = pd.DataFrame(
                self.ordinal_encoder.transform(X[self.ordinal_columns_]),
                columns=self.ordinal_columns_,
                index=X.index,
            )
            parts.append(ord_df)

        if self.nominal_columns_:
            oh = pd.DataFrame(
                self.onehot_encoder.transform(X[self.nominal_columns_]),
                columns=self.onehot_feature_names_,
                index=X.index,
            )
            parts.append(oh)

        if not parts:
            raise ValueError("No feature columns found for transformation.")

        result = pd.concat(parts, axis=1)
        return result.reindex(columns=self.feature_names_)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

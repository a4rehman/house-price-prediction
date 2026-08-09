"""Shared pytest fixtures.

Uses a compact synthetic Ames-like dataset so tests are fast, deterministic,
and do not require downloading the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import Preprocessor
from src.models.training import build_models

QUALITY_LEVELS = ["Ex", "Gd", "TA", "Fa", "Po", "None"]
FENCES = ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
ZONINGS = ["RL", "RM", "FV", "RH", "C", "I", "A", "RP"]
NEIGHBORHOODS = ["NAmes", "CollgCr", "OldTown", "Edwards", "Sawyer", "NridgHt"]
SALE_TYPES = ["WD", "New", "COD", "Con", "ConLD", "ConLI"]
SALE_CONDITIONS = ["Normal", "Abnorml", "Partial", "Family", "Alloca", "AdjLand"]


def make_synthetic_ames(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic dataset with the Ames column schema."""
    rng = np.random.default_rng(seed)
    rows: dict[str, np.ndarray] = {}

    def cat(values: list[str]) -> np.ndarray:
        return rng.choice(values, size=n)

    def cat_na(values: list[str], na_frac: float = 0.05) -> np.ndarray:
        arr = rng.choice(values, size=n).astype(object)
        arr[rng.random(n) < na_frac] = np.nan
        return arr

    def num(lo: float, hi: float) -> np.ndarray:
        return rng.integers(lo, hi, size=n).astype(float)

    rows.update(
        {
            "Id": np.arange(1, n + 1),
            "MSSubClass": num(20, 190),
            "MSZoning": cat(ZONINGS),
            "LotFrontage": rng.choice([*num(21, 110).tolist(), np.nan], size=n).astype(float),
            "LotArea": num(1500, 90000),
            "Street": cat(["Pave", "Grvl"]),
            "Alley": cat_na(["None", "Pave", "Grvl"]),
            "LotShape": cat(["Reg", "IR1", "IR2", "IR3"]),
            "LandContour": cat(["Lvl", "Bnk", "HLS", "Low"]),
            "Utilities": cat(["AllPub", "NoSeWa"]),
            "LotConfig": cat(["Inside", "Corner", "CulDSac", "FR2", "FR3"]),
            "LandSlope": cat(["Gtl", "Mod", "Sev"]),
            "Neighborhood": cat(NEIGHBORHOODS),
            "Condition1": cat(["Norm", "Feedr", "Artery", "PosN", "RRAn"]),
            "Condition2": cat(["Norm", "Feedr", "Artery", "PosN"]),
            "BldgType": cat(["1Fam", "2Fam", "Duplex", "Twnhs", "TwnhsE"]),
            "HouseStyle": cat(["1Story", "2Story", "SLvl", "1.5Fin"]),
            "OverallQual": num(1, 10),
            "OverallCond": num(1, 10),
            "YearBuilt": num(1900, 2010),
            "YearRemodAdd": num(1900, 2010),
            "RoofStyle": cat(["Gable", "Hip", "Flat", "Gambrel"]),
            "RoofMatl": cat(["CompShg", "Metal", "Tar&Grv", "Shingle"]),
            "Exterior1st": cat(["VinylSd", "MetalSd", "HdBoard", "Wd Sdng"]),
            "Exterior2nd": cat(["VinylSd", "MetalSd", "HdBoard", "Plywood"]),
            "MasVnrType": cat_na(["None", "BrkFace", "Stone"]),
            "MasVnrArea": rng.choice([0, 0, 0, 100, 300, 600, np.nan], size=n).astype(float),
            "ExterQual": cat(QUALITY_LEVELS),
            "ExterCond": cat(QUALITY_LEVELS),
            "Foundation": cat(["Poured", "CBlock", "Slab", "Stone"]),
            "BsmtQual": cat_na(["Ex", "Gd", "TA", "Fa", "Po", "None"]),
            "BsmtCond": cat_na(["Ex", "Gd", "TA", "Fa", "Po", "None"]),
            "BsmtExposure": cat_na(["No", "Mn", "Av", "Gd", "None"]),
            "BsmtFinType1": cat_na(["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"]),
            "BsmtFinSF1": num(0, 1600),
            "BsmtFinType2": cat_na(["GLQ", "Unf", "Rec", "LwQ", "None"]),
            "BsmtFinSF2": num(0, 1200),
            "BsmtUnfSF": num(0, 1200),
            "TotalBsmtSF": num(0, 2500),
            "Heating": cat(["GasA", "GasW", "Grav", "Wall"]),
            "HeatingQC": cat(QUALITY_LEVELS),
            "CentralAir": cat(["Y", "N"]),
            "Electrical": cat(["SBrkr", "FuseA", "FuseF", "Mix"]),
            "1stFlrSF": num(400, 2200),
            "2ndFlrSF": num(0, 1800),
            "LowQualFinSF": num(0, 300),
            "GrLivArea": num(500, 4200),
            "BsmtFullBath": num(0, 2),
            "BsmtHalfBath": num(0, 2),
            "FullBath": num(0, 3),
            "HalfBath": num(0, 2),
            "BedroomAbvGr": num(0, 6),
            "KitchenAbvGr": num(0, 3),
            "KitchenQual": cat(QUALITY_LEVELS),
            "TotRmsAbvGrd": num(3, 12),
            "Functional": cat(["Typ", "Min1", "Min2", "Maj1", "Mod", "Maj2"]),
            "Fireplaces": num(0, 3),
            "FireplaceQu": cat_na(["Ex", "Gd", "TA", "Fa", "Po", "None"]),
            "GarageType": cat_na(["Attchd", "Detchd", "BuiltIn", "CarPort", "None"]),
            "GarageYrBlt": rng.choice([*num(1950, 2010).tolist(), np.nan], size=n).astype(float),
            "GarageFinish": cat_na(["Fin", "RFn", "Unf", "None"]),
            "GarageCars": num(0, 4),
            "GarageArea": num(0, 1200),
            "GarageQual": cat(QUALITY_LEVELS),
            "GarageCond": cat(QUALITY_LEVELS),
            "PavedDrive": cat(["Y", "P", "N"]),
            "WoodDeckSF": num(0, 600),
            "OpenPorchSF": num(0, 400),
            "EnclosedPorch": num(0, 300),
            "3SsnPorch": num(0, 300),
            "ScreenPorch": num(0, 300),
            "PoolArea": num(0, 300),
            "PoolQC": cat_na(["Ex", "Gd", "TA", "Fa", "None"]),
            "Fence": cat_na(FENCES),
            "MiscFeature": cat_na(["None", "Shed", "Othr"]),
            "MiscVal": num(0, 1000),
            "MoSold": num(1, 13),
            "YrSold": num(2006, 2011),
            "SaleType": cat(SALE_TYPES),
            "SaleCondition": cat(SALE_CONDITIONS),
            "SalePrice": num(60000, 600000).astype(float),
        }
    )
    df = pd.DataFrame(rows)
    return df


@pytest.fixture(scope="session")
def ames_synthetic() -> pd.DataFrame:
    return make_synthetic_ames(n=300, seed=7)


@pytest.fixture()
def tiny_artifacts(ames_synthetic, tmp_path_factory):
    """Fit a tiny model + preprocessor and persist to a temp directory."""
    tmp_path = tmp_path_factory.mktemp("model")
    df = ames_synthetic.copy()
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    pre = Preprocessor(scale=True).fit(X, y)
    Xt = pre.transform(X)

    model = build_models(random_state=0)["RandomForest"]
    model.set_params(n_estimators=10, max_depth=4)
    import numpy as np

    model.fit(Xt, np.log1p(y))

    from src.models.registry import save_local_artifacts

    save_local_artifacts(model, pre, {"model_name": "RandomForest_test"}, tmp_path)
    return tmp_path

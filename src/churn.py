"""Customer churn modelling, evaluation, and inference services.

The module is deliberately self-contained so the app can run as a Hugging Face
Space with no private dataset.  Upload a CSV with a ``Churn`` column to train
on production data; until then a realistic deterministic demo cohort is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "tenure", "monthly_charges", "total_charges", "support_tickets",
    "contract", "internet_service", "payment_method", "senior_citizen",
    "paperless_billing",
]
NUMERIC = [
    "tenure", "monthly_charges", "total_charges", "support_tickets",
    "senior_citizen",
]
CATEGORICAL = [x for x in FEATURES if x not in NUMERIC]


def demo_data(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generate a representative telco cohort for a runnable first launch."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    monthly = np.clip(rng.normal(72, 28, rows), 18, 145).round(2)
    tickets = rng.poisson(1.4, rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], rows, p=[.56, .25, .19],
    )
    internet = rng.choice(
        ["Fiber optic", "DSL", "None"], rows, p=[.46, .43, .11],
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        rows,
    )
    senior = rng.binomial(1, .17, rows)
    paperless = rng.choice(["Yes", "No"], rows, p=[.6, .4])
    logit = (
        -1.5
        + 1.25 * (contract == "Month-to-month")
        + .72 * (internet == "Fiber optic")
        + .42 * (payment == "Electronic check")
        + .18 * tickets
        + .48 * senior
        - .028 * tenure
        + .004 * (monthly - 70)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    total = (monthly * np.maximum(tenure, 1) * rng.uniform(.9, 1.1, rows)).round(2)
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, rows + 1)],
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets": tickets,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "paperless_billing": paperless,
        "Churn": churn,
    })


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise friendly CSV headers and guarantee the model feature contract."""
    clean = df.copy()
    clean.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in clean.columns
    ]
    aliases = {
        "monthlycharges": "monthly_charges",
        "totalcharges": "total_charges",
        "supporttickets": "support_tickets",
        "seniorcitizen": "senior_citizen",
        "paperlessbilling": "paperless_billing",
        "customerid": "customer_id",
        "churn": "Churn",
    }
    clean = clean.rename(columns={k: v for k, v in aliases.items() if k in clean.columns})
    defaults: dict[str, Any] = {
        "tenure": 12,
        "monthly_charges": 70.,
        "total_charges": 840.,
        "support_tickets": 0,
        "contract": "Month-to-month",
        "internet_service": "DSL",
        "payment_method": "Electronic check",
        "senior_citizen": 0,
        "paperless_billing": "Yes",
    }
    for col, value in defaults.items():
        if col not in clean:
            clean[col] = value
    for col in NUMERIC:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean


def build_pipeline(model: Any) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("select", SelectPercentile(mutual_info_classif, percentile=85)),
        ("model", model),
    ])


@dataclass
class ChurnService:
    artifact_path: Path
    pipeline: Pipeline | None = None
    metrics: dict[str, Any] | None = None

    def load(self) -> ChurnService:
        if self.artifact_path.exists():
            saved = joblib.load(self.artifact_path)
            self.pipeline, self.metrics = saved["pipeline"], saved["metrics"]
        return self

    def train(self, frame: pd.DataFrame | None = None) -> dict[str, Any]:
        df = prepare(frame if frame is not None else demo_data())
        target = df.get("Churn")
        if target is None:
            raise ValueError("Training data must include a Churn column.")
        y = target.astype(str).str.lower().isin(["1", "yes", "true", "churned"]).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            df[FEATURES], y, test_size=.22, random_state=42, stratify=y,
        )
        candidates = {
            "Logistic Regression": LogisticRegression(
                max_iter=1500, class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        scores: dict[str, float] = {}
        fitted: dict[str, Pipeline] = {}
        for name, model in candidates.items():
            pipe = build_pipeline(model)
            pipe.fit(x_train, y_train)
            scores[name] = float(roc_auc_score(y_test, pipe.predict_proba(x_test)[:, 1]))
            fitted[name] = pipe
        winner = max(scores, key=scores.get)  # type: ignore[arg-type]
        # Tune the winner using a compact, deployment-friendly search.
        if winner == "Random Forest":
            grid = {
                "model__n_estimators": [160, 280],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3],
            }
            search = GridSearchCV(
                build_pipeline(candidates[winner]), grid,
                scoring="roc_auc", cv=3, n_jobs=-1,
            )
            search.fit(x_train, y_train)
            self.pipeline = search.best_estimator_
        else:
            self.pipeline = fitted[winner]
        prob = self.pipeline.predict_proba(x_test)[:, 1]
        pred = (prob >= .5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        self.metrics = {
            "model": winner,
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
            "average_precision": round(float(average_precision_score(y_test, prob)), 4),
            "comparison": {k: round(v, 4) for k, v in scores.items()},
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "precision_recall": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
            },
            "classification_report": classification_report(y_test, pred, output_dict=True),
            "records": len(df),
            "churn_rate": round(float(y.mean()), 4),
        }
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics}, self.artifact_path)
        return self.metrics

    def predict(self, customer: dict[str, Any]) -> dict[str, Any]:
        if self.pipeline is None:
            self.load()
        if self.pipeline is None:
            self.train()
        row = prepare(pd.DataFrame([customer]))
        probability = float(self.pipeline.predict_proba(row[FEATURES])[:, 1][0])
        risk = "High" if probability >= .7 else "Medium" if probability >= .35 else "Low"
        recommendations = {
            "High": [
                "Assign a retention specialist within 24 hours",
                "Offer a tailored plan review or loyalty incentive",
                "Prioritize support-ticket resolution",
            ],
            "Medium": [
                "Send a proactive value and usage check-in",
                "Promote annual-contract savings",
                "Monitor support experience",
            ],
            "Low": [
                "Maintain engagement through value communications",
                "Invite customer feedback",
                "Review risk on the next billing cycle",
            ],
        }[risk]
        return {
            "customer_id": str(customer.get("customer_id", "NEW-CUSTOMER")),
            "churn_probability": round(probability, 4),
            "risk_category": risk,
            "recommendations": recommendations,
        }

    def explain(self, customer: dict[str, Any]) -> list[dict[str, Any]]:
        row = prepare(pd.DataFrame([customer])).iloc[0]
        baseline = {"tenure": 36, "monthly_charges": 70, "support_tickets": 1}
        values = []
        for col in [
            "contract", "internet_service", "payment_method",
            "tenure", "support_tickets", "monthly_charges",
        ]:
            val = row[col]
            impact = 0.0
            if col == "contract" and val == "Month-to-month":
                impact = .25
            elif col == "internet_service" and val == "Fiber optic":
                impact = .14
            elif col == "payment_method" and val == "Electronic check":
                impact = .10
            elif col == "tenure":
                impact = (baseline[col] - float(val)) / 180
            elif col == "support_tickets":
                impact = (float(val) - baseline[col]) / 10
            elif col == "monthly_charges":
                impact = (float(val) - baseline[col]) / 350
            values.append({"feature": col, "value": str(val), "impact": round(impact, 3)})
        return sorted(values, key=lambda item: abs(item["impact"]), reverse=True)

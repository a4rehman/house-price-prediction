"""API integration tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, app_state
from src.models.inference import PredictionService


@pytest.fixture()
def client(tiny_artifacts):
    app_state.service = PredictionService(local_dir=tiny_artifacts)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "endpoints" in resp.json()


def test_single_predict(client):
    payload = {"OverallQual": 7, "GrLivArea": 1800, "LotArea": 9000}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price"] > 0
    assert body["model"]


def test_single_predict_with_explanation(client):
    payload = {"OverallQual": 8, "GrLivArea": 2500, "Neighborhood": "NridgHt"}
    resp = client.post("/api/v1/predict", json=payload)
    body = resp.json()
    assert body["predicted_price"] > 0
    assert body["explanation"] is not None


def test_batch_predict(client):
    payload = [
        {"OverallQual": 6, "GrLivArea": 1200},
        {"OverallQual": 9, "GrLivArea": 3200},
    ]
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_csv(client, ames_synthetic):
    df = ames_synthetic.head(5).drop(columns=["SalePrice"])
    resp = client.post(
        "/api/v1/predict/csv",
        files={"file": ("sample.csv", df.to_csv(index=False), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 5


def test_predict_csv_invalid(client):
    resp = client.post(
        "/api/v1/predict/csv",
        files={"file": ("bad.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400


def test_explain_endpoint(client):
    resp = client.post(
        "/api/v1/explain", json={"OverallQual": 7, "GrLivArea": 2000}
    )
    assert resp.status_code == 200
    assert "predicted_price" in resp.json()


def test_models_endpoint(client):
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "active" in body
    assert "registered" in body

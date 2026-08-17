---
title: House Price Prediction Platform
emoji: "🏠"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
hardware: cpu-basic
---

# 🏠 House Price Prediction Platform

A production-grade, end-to-end Machine Learning platform for predicting residential home sale prices using the Ames Housing Dataset.

---

## 🌟 Highlights

- **Complete ML Pipeline**: Automated data ingestion, EDA generation, outlier cleaning, cross-validated model selection, and hyperparameter tuning with Optuna.
- **Model Diversity**: Evaluates Ridge, Lasso, ElasticNet, DecisionTree, ExtraTrees, RandomForest, GradientBoosting, XGBoost, LightGBM, and CatBoost.
- **Explainability**: Computes SHAP global feature importances and local per-prediction waterfall explanations.
- **Dual Interfaces**:
  - **Gradio Space**: Deployed to Hugging Face Spaces with single and batch CSV valuation interfaces.
  - **Streamlit Dashboard**: Rich multi-page interactive web UI with live exploratory data analysis and model comparison views.
- **FastAPI Backend**: Async REST API supporting single, batch, CSV predictions, and model metadata endpoints.
- **Production CI/CD**: GitHub Actions workflows for continuous linting (Ruff), testing (Pytest), model training validation, and container image publishing.

---

## 🧱 Tech Stack

**Python 3.11+ · Scikit-learn · XGBoost · LightGBM · CatBoost · Pandas · NumPy · FastAPI · Gradio · Streamlit · Docker · MLflow · Optuna · SHAP · GitHub Actions**

---

## 📂 Project Structure

```
house_price_predicition/
├── .github/workflows/        # CI + CD pipelines
├── app.py                    # Gradio app entrypoint for Hugging Face Space
├── data/
│   ├── raw/                  # downloaded dataset (git-ignored)
│   └── processed/            # cleaned train/test splits
├── src/
│   ├── config.py             # pydantic-settings configuration
│   ├── logging_config.py     # structured JSON logging
│   ├── utils.py              # shared helpers
│   ├── pipeline.py           # end-to-end training orchestrator
│   ├── data/
│   │   ├── loader.py         # download + load
│   │   ├── eda.py            # EDA report generation
│   │   └── preprocessing.py  # missing values, outliers, encoding, scaling
│   ├── features/
│   │   └── engineering.py    # domain feature engineering
│   ├── models/
│   │   ├── training.py       # model zoo, CV, final training
│   │   ├── comparison.py     # leaderboard + diagnostics
│   │   ├── tuning.py         # Optuna hyperparameter optimisation
│   │   ├── explainability.py # SHAP global + local explanations
│   │   ├── inference.py      # PredictionService (shared by API/dashboard)
│   │   ├── metrics.py        # RMSE / MAE / R² / log-space metrics
│   │   └── registry.py       # MLflow + local artifact registry
│   ├── api/                  # FastAPI application & endpoints
│   └── dashboard/            # Streamlit dashboard
├── scripts/                  # CLI entrypoints
├── notebooks/                # EDA notebook
├── tests/                    # pytest suite
├── artifacts/                # models, plots, reports (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml            # ruff + pytest config
```

---

## 🚀 Quickstart

### 1. Clone & Set Up

```bash
git clone https://github.com/a4rehman/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Download Data & Run Training

```bash
python scripts/download_data.py
python scripts/run_training.py
```

### 3. Run Gradio / Streamlit / API

```bash
# Gradio HF Space App
python app.py

# Streamlit Dashboard
python scripts/run_dashboard.py

# FastAPI REST API
python scripts/run_api.py
```

---

## 🌐 REST API

Base URL: `http://localhost:8000` — interactive docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service + model health |
| `GET` | `/api/v1/models` | Registered model versions & active model |
| `POST` | `/api/v1/predict` | Single-house prediction **+ SHAP explanation** |
| `POST` | `/api/v1/predict/batch` | Predict a JSON array of houses |
| `POST` | `/api/v1/predict/csv` | Upload a CSV of houses |
| `POST` | `/api/v1/explain` | SHAP explanation for one house |

---

## ✅ Testing & Linting

```bash
ruff check src tests scripts app.py
pytest --cov=src --cov-report=term-missing
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

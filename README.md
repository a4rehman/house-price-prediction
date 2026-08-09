# 🏠 House Price Prediction Platform

A **production-ready machine-learning platform** that predicts residential
sale prices (Ames, Iowa dataset) and serves those predictions through a REST
API and an interactive dashboard.

Built like a real ML product: **EDA → feature engineering → model comparison →
hyperparameter tuning → SHAP explainability → versioned registry → containerised
API/dashboard → CI/CD → tests → documentation**.

---

## ✨ Features

| Area | What's included |
|---|---|
| **EDA** | Auto-generated plots, JSON summary, HTML report (`src/data/eda.py`) |
| **Feature engineering** | Domain features: `TotalSF`, `TotalBath`, `HouseAge`, flags, composite quality scores |
| **Missing values** | Domain-aware imputation (median-by-neighborhood, "None" category, 0-fill) |
| **Outliers** | IQR winsorisation + removal of known bad records |
| **Encoding** | Explicit ordinal rankings + one-hot for nominal, `handle_unknown='ignore'` |
| **Scaling** | `StandardScaler` via a fitted, reusable `Preprocessor` |
| **Model comparison** | 9 models, k-fold CV leaderboard + comparison plot |
| **Hyperparameter tuning** | Optuna (TPE sampler) on the winning family |
| **Explainability** | SHAP summary, bar, and waterfall plots; API `/explain` |
| **Prediction dashboard** | Streamlit — single, batch (CSV), insights, model comparison |
| **REST API** | FastAPI — single / batch / CSV upload + SHAP + registry endpoints |
| **Batch prediction** | CLI (`scripts/predict_csv.py`) and CSV upload endpoint |
| **Model versioning** | MLflow registry + local artifacts with metadata |
| **Logging** | Structured JSON logs, console + file |
| **Deployment** | Multi-stage-ready `Dockerfile` + `docker-compose.yml` |
| **CI/CD** | GitHub Actions — lint, test, train, publish image |
| **Tests** | `pytest` suite covering every layer |

---

## 🧱 Tech Stack

**Python 3.11+ · Scikit-learn · XGBoost · LightGBM · CatBoost · Pandas · NumPy ·
FastAPI · Streamlit · Docker · MLflow · Optuna · SHAP · GitHub Actions**

---

## 📂 Project Structure

```
house_price_predicition/
├── .github/workflows/        # CI + CD pipelines
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

### 1. Clone & set up

```bash
git clone <repo-url> house-price && cd house-price
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests / lint
```

### 2. Download the data & run EDA

```bash
python scripts/download_data.py
python scripts/run_eda.py
# Output: artifacts/plots/*.png, artifacts/reports/eda_summary.json
```

### 3. Train, compare, tune & register the model

```bash
python scripts/run_training.py
```

This will:
1. Clean outliers and split the data
2. Cross-validate 9 model families
3. Tune the winner with Optuna
4. Train the final model, evaluate on a hold-out set
5. Compute SHAP global importance
6. Persist artifacts to `artifacts/models/best_model/`
7. Log the run to MLflow (`mlruns/`)

> Training runs several model families and the full Optuna study, so it can
> take a few minutes. Use `--no-tune` for a fast smoke run.

### 4. Serve the REST API

```bash
python scripts/run_api.py
# or
uvicorn src.api.main:app --reload
```

Open the interactive docs at **http://localhost:8000/docs**.

### 5. Open the prediction dashboard

```bash
python scripts/run_dashboard.py
# or
streamlit run src/dashboard/app.py
```

Open **http://localhost:8501**.

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

### Example — single prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "OverallCond": 6,
    "GrLivArea": 1800,
    "TotalBsmtSF": 900,
    "LotArea": 9500,
    "YearBuilt": 2005,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "HalfBath": 1,
    "Neighborhood": "NAmes",
    "CentralAir": "Y",
    "GarageCars": 2
  }'
```

Response:

```json
{
  "id": 1,
  "predicted_price": 215043.5,
  "prediction": 215043.5,
  "model": "XGBoost_tuned",
  "explanation": {
    "base_value": 11.9,
    "expected_value": 11.9,
    "explanation": [
      { "feature": "GrLivArea", "value": 0.32 },
      { "feature": "OverallQual", "value": 0.18 }
    ]
  }
}
```

> Every attribute in the schema is **optional** — anything missing is imputed by
> the fitted preprocessor, so minimal payloads like `{"OverallQual": 8}` work.

### Example — CSV upload

```bash
curl -X POST http://localhost:8000/api/v1/predict/csv \
  -F "file=@data/processed/test.csv"
```

### Batch prediction from the CLI

```bash
python scripts/predict_csv.py data/processed/test.csv -o predictions.csv
```

---

## 📊 Streamlit Dashboard

- **Single Prediction** — interactive form + live SHAP waterfall
- **Batch Prediction** — upload a CSV, download results
- **Data Insights** — EDA plots rendered from the pipeline output
- **Model Comparison** — CV leaderboard, hold-out diagnostics, run summary

---

## 🐳 Docker

### Build & run the API

```bash
docker build -t house-price:latest .
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" house-price:latest
```

### Full stack (API + dashboard + MLflow tracking server)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |

---

## 📈 Model Versioning with MLflow

Every training run is logged to `mlruns/` (local file store by default) with:

- Parameters (best model family + tuned hyperparameters)
- Metrics (hold-out RMSE/MAE/R² and CV RMSE on log scale)
- Artifacts (plots, reports, and the model registered as `house_price_predictor`)

```bash
# Browse experiments
mlflow ui --port 5000

# List registered versions via the API
curl http://localhost:8000/api/v1/models
```

Set `MODEL_URI=mlflow://models:/house_price_predictor/Production` in your
environment to serve a specific registered version instead of the local
artifact.

---

## ✅ Tests

```bash
pytest                                   # full suite
pytest --cov=src --cov-report=term-missing   # with coverage
ruff check src tests scripts              # lint
```

The suite uses a synthetic Ames-like dataset — it is fast and requires no
network access.

---

## 🤖 CI/CD

- **CI** (`.github/workflows/ci.yml`): on every push/PR — ruff lint, `pytest`
  on Python 3.11 & 3.12, then a real training run + API smoke test, uploading
  model artifacts.
- **CD** (`.github/workflows/cd.yml`): on `main` and version tags — builds and
  publishes the Docker image to **GHCR**.

---

## 🧪 Configuration

All settings live in `src/config.py` and can be overridden via environment
variables or a `.env` file (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `DATA_URL` | *inria mirror* | Raw dataset URL |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `N_TRIALS` | `30` | Optuna optimisation trials |
| `TEST_SIZE` | `0.2` | Hold-out fraction |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `REGISTERED_MODEL_NAME` | `house_price_predictor` | MLflow registry name |
| `MODEL_URI` | *(empty)* | Serve a registered MLflow model |
| `API_PORT` | `8000` | API port |
| `DASHBOARD_PORT` | `8501` | Dashboard port |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- Ames housing data (De Cock 2011), mirrored by the
  [scikit-learn MOOC](https://github.com/inria/scikit-learn-mooc).
- The Kaggle *House Prices: Advanced Regression Techniques* competition.

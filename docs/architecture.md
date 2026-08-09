# Architecture

## Overview

```
                     ┌────────────────────┐
                     │   data / raw CSV    │
                     └─────────┬──────────┘
                               ▼
               ┌───────────────────────────────┐
               │        Data layer             │
               │  loader · eda · preprocessing │
               │  (impute · outliers · encode) │
               └──────────────┬────────────────┘
                              ▼
               ┌───────────────────────────────┐
               │       Feature layer           │
               │      engineering.py           │
               └──────────────┬────────────────┘
                              ▼
               ┌───────────────────────────────┐
               │        Model layer            │
               │  training · comparison ·      │
               │  tuning (Optuna) · SHAP       │
               └───────┬───────────────┬───────┘
                       │               │
                       ▼               ▼
              ┌──────────────┐  ┌──────────────┐
              │   Local      │  │   MLflow     │
              │   artifacts  │  │   registry   │
              └──────┬───────┘  └──────┬───────┘
                     └───────┬──────────┘
                             ▼
                  ┌──────────────────────┐
                  │   PredictionService  │   (shared inference)
                  └───────┬──────────────┘
                          ▼
              ┌───────────┴───────────┐
              ▼                       ▼
      ┌────────────────┐      ┌────────────────┐
      │   FastAPI      │      │  Streamlit     │
      │   REST API     │      │  Dashboard     │
      └────────────────┘      └────────────────┘
```

## Key design decisions

1. **One fitted preprocessor everywhere.** The `Preprocessor` is fitted once
   on training data and reused for validation, test, batch, and single-record
   inference. The feature space is therefore identical at training and serving
   time — the number-one cause of train/serve skew is eliminated.

2. **Log-target regression.** Models fit on `log1p(SalePrice)`. This stabilises
   the variance of the (right-skewed) target and is standard practice for this
   dataset. Predictions are exponentiated back at serving time.

3. **Explicit ordinal encodings.** Ames has many ordered qualities
   (`Ex > Gd > TA > Fa > Po`). Mapping them with a curated rank vector rather
   than one-hot encoding preserves order and keeps feature count manageable.

4. **Robust categorical serving.** One-hot encoding uses
   `handle_unknown="ignore"` so a brand-new category at inference time never
   crashes the pipeline — it simply contributes no signal.

5. **Shared inference layer.** `PredictionService` is the single class used by
   both the API and the dashboard. It can load from local artifacts or from an
   MLflow-registered version (`MODEL_URI`), which makes A/B testing and model
   rollback trivial.

6. **Structured logging.** All components emit JSON logs through one configured
   logger, with run-level context attached via a logging `Filter`.

## Serving paths

- **REST API** (`src/api/main.py`): FastAPI app with CORS, lazy model loading,
  `POST /api/v1/predict`, `/predict/batch`, `/predict/csv`, `/explain`, plus
  health and registry introspection.
- **Dashboard** (`src/dashboard/app.py`): Streamlit multi-page app sharing the
  same `PredictionService`.
- **Batch** (`scripts/predict_csv.py`): offline CSV → predictions via the same
  service.

## Deployment

- `Dockerfile` builds a slim Python 3.12 image with all boosting backends.
- `docker-compose.yml` runs the API, dashboard, and an MLflow tracking server
  with SQLite backend.
- CI runs lint + tests + a real training run; CD publishes the image to GHCR.

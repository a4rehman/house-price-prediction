---
title: Customer Churn Intelligence
emoji: "🎯"
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Customer Churn Intelligence

An enterprise-grade churn prediction system with a modern Gradio dashboard and a FastAPI service. It is immediately runnable using a deterministic demo customer cohort, and can be retrained on an uploaded labelled customer CSV.

## What’s included

- Data cleaning, missing-value handling, categorical encoding and feature selection
- EDA-ready demo dataset and feature contract
- Logistic Regression, Random Forest and Gradient Boosting model comparison
- Hyperparameter tuning for the selected Random Forest family
- ROC-AUC, average precision, ROC/precision-recall curve data, and confusion matrix
- Local customer-level feature-impact explanations and retention recommendations
- Customer risk dashboard, probability display, risk categories and action recommendations
- FastAPI prediction, batch, CSV, metrics, explanation, and admin training endpoints
- Optional API-key authentication using `API_KEY`
- Docker Compose and GitHub Actions configuration

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The FastAPI service runs with:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `/docs` for interactive API documentation. When `API_KEY` is configured, pass it as the `X-API-Key` request header.

## Train on your customer data

Use `POST /api/v1/admin/train` with a CSV containing `Churn` and these inputs:

`customer_id`, `tenure`, `monthly_charges`, `total_charges`, `support_tickets`, `contract`, `internet_service`, `payment_method`, `senior_citizen`, `paperless_billing`.

The target accepts values such as `1/0`, `Yes/No`, or `True/False`.

## Hugging Face Space

This repository is Space-ready. Commit `app.py`, `requirements.txt`, and this README, then push to your Space repository. Put `API_KEY` in the Space’s Secrets settings before exposing the API publicly.

Never commit access tokens or credentials. If a token was pasted into a chat or terminal, revoke it and generate a replacement.

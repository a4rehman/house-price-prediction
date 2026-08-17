"""Gradio Space entry point for House Price Prediction Platform."""
from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from src.config import settings
from src.data.loader import download_raw_data
from src.logging_config import setup_logging
from src.models.inference import PredictionService
from src.pipeline import run_training

setup_logging(log_file=None)


def ensure_model_ready() -> PredictionService:
    service = PredictionService()
    model_file = settings.models_dir / "best_model" / "model.joblib"
    if not model_file.exists():
        try:
            download_raw_data(force=False)
            run_training(tune=False, register_mlflow=False)
        except Exception:
            pass
    try:
        service.load()
    except Exception:
        pass
    return service


service = ensure_model_ready()


def predict_price(
    overall_qual: int,
    overall_cond: int,
    gr_liv_area: float,
    total_bsmt_sf: float,
    year_built: int,
    bedroom_abv_gr: int,
    full_bath: int,
    half_bath: int,
    lot_area: float,
    neighborhood: str,
    exter_qual: str,
    kitchen_qual: str,
    fireplace_qu: str,
    central_air: str,
    garage_cars: int,
) -> tuple[str, str, pd.DataFrame]:
    payload: dict[str, Any] = {
        "OverallQual": int(overall_qual),
        "OverallCond": int(overall_cond),
        "GrLivArea": float(gr_liv_area),
        "TotalBsmtSF": float(total_bsmt_sf),
        "YearBuilt": int(year_built),
        "BedroomAbvGr": int(bedroom_abv_gr),
        "FullBath": int(full_bath),
        "HalfBath": int(half_bath),
        "LotArea": float(lot_area),
        "Neighborhood": str(neighborhood),
        "ExterQual": str(exter_qual),
        "KitchenQual": str(kitchen_qual),
        "FireplaceQu": str(fireplace_qu),
        "CentralAir": str(central_air),
        "GarageCars": int(garage_cars),
    }

    try:
        res = service.predict_single(payload)
        price = res.get("predicted_price", 0.0)
        formatted_price = f"## 💰 Estimated Valuation: **${price:,.2f}**"
        model_name = f"**Model:** {res.get('model', 'Ensemble / Regressor')}"

        explanation = res.get("explanation", {})
        shap_items = explanation.get("explanation", [])
        if shap_items:
            df_factors = pd.DataFrame(shap_items)
            df_factors = df_factors.rename(
                columns={"feature": "Feature", "value": "SHAP Impact ($)"}
            )
        else:
            df_factors = pd.DataFrame(
                [{"Feature": k, "Input Value": str(v)} for k, v in payload.items()]
            )
        return formatted_price, model_name, df_factors
    except Exception as exc:
        return f"## ⚠️ Error in prediction: {exc}", "", pd.DataFrame()


def predict_batch_csv(file_obj: Any) -> tuple[str, pd.DataFrame | None]:
    if file_obj is None:
        return "Please upload a CSV file.", None
    try:
        df = pd.read_csv(file_obj.name if hasattr(file_obj, "name") else file_obj)
        preds = service.predict(df)
        summary = f"Successfully predicted {len(preds):,} properties."
        return summary, preds
    except Exception as exc:
        return f"Error processing CSV: {exc}", None


def get_model_summary() -> tuple[dict[str, Any], pd.DataFrame]:
    meta = getattr(service, "metadata", {})
    holdout = meta.get("holdout", {})
    metrics_rows = [
        {"Metric": "Model Architecture", "Value": str(meta.get("model_name", "Tuned Model"))},
        {"Metric": "CV RMSE (log)", "Value": str(meta.get("cv_rmse_log", "N/A"))},
        {"Metric": "Holdout RMSE ($)", "Value": f"${holdout.get('rmse', 0):,.2f}" if 'rmse' in holdout else "N/A"},
        {"Metric": "Holdout MAE ($)", "Value": f"${holdout.get('mae', 0):,.2f}" if 'mae' in holdout else "N/A"},
        {"Metric": "Holdout R²", "Value": str(holdout.get("r2", "N/A"))},
    ]
    return meta, pd.DataFrame(metrics_rows)


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="House Price Prediction Platform") as demo:
    gr.Markdown(
        "# 🏠 House Price Prediction Platform\n"
        "### Production-Grade Ames Housing Valuation & Explainable ML"
    )

    with gr.Tabs():
        with gr.Tab("🔮 Single Property Valuation"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        overall_qual = gr.Slider(1, 10, value=7, step=1, label="Overall Quality (1-10)")
                        overall_cond = gr.Slider(1, 10, value=5, step=1, label="Overall Condition (1-10)")
                    with gr.Row():
                        gr_liv_area = gr.Number(value=1800, label="Above Ground Living Area (sq ft)")
                        total_bsmt_sf = gr.Number(value=1000, label="Total Basement Area (sq ft)")
                        lot_area = gr.Number(value=9500, label="Lot Area (sq ft)")
                    with gr.Row():
                        year_built = gr.Slider(1870, 2025, value=2005, step=1, label="Year Built")
                        bedroom_abv_gr = gr.Slider(0, 8, value=3, step=1, label="Bedrooms Above Grade")
                        full_bath = gr.Slider(0, 5, value=2, step=1, label="Full Bathrooms")
                        half_bath = gr.Slider(0, 3, value=1, step=1, label="Half Bathrooms")
                    with gr.Row():
                        neighborhood = gr.Dropdown(
                            [
                                "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                                "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
                                "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
                                "IDOTRR", "ClearCr", "StoneBr", "SWISU", "Blmngtn",
                            ],
                            value="CollgCr",
                            label="Neighborhood",
                        )
                        kitchen_qual = gr.Dropdown(["Ex", "Gd", "TA", "Fa", "Po"], value="Gd", label="Kitchen Quality")
                        exter_qual = gr.Dropdown(["Ex", "Gd", "TA", "Fa", "Po"], value="Gd", label="Exterior Quality")
                    with gr.Row():
                        fireplace_qu = gr.Dropdown(["Ex", "Gd", "TA", "Fa", "Po", "None"], value="Gd", label="Fireplace Quality")
                        central_air = gr.Radio(["Y", "N"], value="Y", label="Central Air")
                        garage_cars = gr.Slider(0, 5, value=2, step=1, label="Garage Capacity (Cars)")

                    predict_btn = gr.Button("Calculate Valuation", variant="primary")

                with gr.Column(scale=1):
                    price_output = gr.Markdown("## Click 'Calculate Valuation' to predict")
                    model_output = gr.Markdown("")
                    factors_output = gr.Dataframe(label="Feature Impact / Explanations", interactive=False)

            predict_btn.click(
                predict_price,
                inputs=[
                    overall_qual, overall_cond, gr_liv_area, total_bsmt_sf,
                    year_built, bedroom_abv_gr, full_bath, half_bath,
                    lot_area, neighborhood, exter_qual, kitchen_qual,
                    fireplace_qu, central_air, garage_cars,
                ],
                outputs=[price_output, model_output, factors_output],
            )

        with gr.Tab("📁 Batch CSV Prediction"):
            gr.Markdown("Upload a CSV file containing property features to run bulk valuations.")
            csv_input = gr.File(label="Upload Ames CSV File", file_types=[".csv"])
            batch_btn = gr.Button("Run Batch Predictions", variant="primary")
            batch_status = gr.Markdown()
            batch_output = gr.Dataframe(label="Batch Predictions", interactive=False)
            batch_btn.click(predict_batch_csv, inputs=[csv_input], outputs=[batch_status, batch_output])

        with gr.Tab("📊 Model Metadata & Metrics"):
            meta_btn = gr.Button("Refresh Model Info")
            meta_kpis = gr.Dataframe(label="Key Performance Indicators", interactive=False)
            meta_json = gr.JSON(label="Full Model Metadata")
            meta_btn.click(get_model_summary, outputs=[meta_json, meta_kpis])

demo.launch(ssr_mode=False)

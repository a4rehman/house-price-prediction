"""Gradio Space entry point for Customer Churn Intelligence."""
from __future__ import annotations

import gradio as gr
import pandas as pd
from src.churn import ChurnService, demo_data  # noqa: F401
from src.config import settings

service = ChurnService(settings.models_dir / "churn_model.joblib").load()
if service.pipeline is None:
    service.train()


def assess(
    customer_id, tenure, monthly, total, tickets,
    contract, internet, payment, senior, paperless,
):
    payload = {
        "customer_id": customer_id,
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets": tickets,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "paperless_billing": paperless,
    }
    result = service.predict(payload)
    factors = service.explain(payload)
    gauge = f"## {result['churn_probability']:.0%}\n### {result['risk_category']} risk"
    recommendations = "\n".join(f"- {item}" for item in result["recommendations"])
    return gauge, result["risk_category"], recommendations, pd.DataFrame(factors)


def admin_summary():
    metrics = service.metrics or service.train()
    kpis = pd.DataFrame([
        {"Metric": "ROC-AUC", "Value": metrics["roc_auc"]},
        {"Metric": "Average precision", "Value": metrics["average_precision"]},
        {"Metric": "Training records", "Value": metrics["records"]},
        {"Metric": "Churn rate", "Value": f"{metrics['churn_rate']:.1%}"},
    ])
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual retained", "Actual churned"],
        columns=["Predicted retained", "Predicted churned"],
    )
    return metrics, kpis, matrix


theme = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="Customer Churn Intelligence") as demo:
    gr.Markdown(
        "# Customer Churn Intelligence\n"
        "### Enterprise retention decisions, made actionable."
    )
    with gr.Tabs():
        with gr.Tab("Customer Search & Risk"):
            with gr.Row():
                with gr.Column(scale=2):
                    customer_id = gr.Textbox(value="CUST-10001", label="Customer ID")
                    with gr.Row():
                        tenure = gr.Slider(0, 120, value=8, step=1, label="Tenure (months)")
                        monthly = gr.Slider(0, 200, value=92, label="Monthly charges")
                    with gr.Row():
                        total = gr.Number(value=736, label="Total charges")
                        tickets = gr.Slider(0, 15, value=2, step=1, label="Support tickets")
                    with gr.Row():
                        contract = gr.Dropdown(
                            ["Month-to-month", "One year", "Two year"],
                            value="Month-to-month", label="Contract",
                        )
                        internet = gr.Dropdown(
                            ["Fiber optic", "DSL", "None"],
                            value="Fiber optic", label="Internet service",
                        )
                    with gr.Row():
                        payment = gr.Dropdown(
                            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                            value="Electronic check", label="Payment method",
                        )
                        senior = gr.Radio([0, 1], value=0, label="Senior citizen")
                    paperless = gr.Radio(["Yes", "No"], value="Yes", label="Paperless billing")
                    run = gr.Button("Assess churn risk", variant="primary")
                with gr.Column():
                    gauge = gr.Markdown()
                    risk = gr.Textbox(label="Risk category")
                    recs = gr.Markdown(label="Recommended next actions")
            factors = gr.Dataframe(label="Explainability: drivers of risk", interactive=False)
            run.click(
                assess,
                [customer_id, tenure, monthly, total, tickets,
                 contract, internet, payment, senior, paperless],
                [gauge, risk, recs, factors],
            )
        with gr.Tab("Admin Dashboard"):
            refresh = gr.Button("Refresh model performance")
            overview = gr.JSON(label="Model analytics")
            kpis = gr.Dataframe(label="Key metrics", interactive=False)
            matrix = gr.Dataframe(label="Confusion matrix", interactive=False)
            refresh.click(admin_summary, outputs=[overview, kpis, matrix])
        with gr.Tab("Data & API"):
            gr.Markdown(
                "Upload labelled customer data to `POST /api/v1/admin/train`, then use "
                "the authenticated prediction API. Set `API_KEY` in your Space secrets "
                "to enforce API-key authentication.\n\n"
                "Expected fields: `customer_id`, `tenure`, `monthly_charges`, "
                "`total_charges`, `support_tickets`, `contract`, `internet_service`, "
                "`payment_method`, `senior_citizen`, `paperless_billing`, and "
                "`Churn` for training."
            )

demo.launch()

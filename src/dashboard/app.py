"""House Price Prediction Dashboard (Streamlit).

Features:
  * Single-house prediction with live SHAP waterfall explanation
  * CSV batch prediction with download
  * EDA visualisations and model comparison
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import __version__
from ..config import settings
from ..data.eda import run_eda
from ..data.loader import load_raw_data
from ..logging_config import setup_logging
from ..models.inference import PredictionService

setup_logging(log_file=None)
st.set_page_config(page_title="House Price Prediction", layout="wide", page_icon="🏠")

PLOTS_DIR = settings.plots_dir


# ---------------------------------------------------------------------------
# Service loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_service() -> PredictionService:
    return PredictionService().load()


def load_eda(df: pd.DataFrame) -> dict:
    """Cached EDA run used for the insights page."""
    key = df.shape
    if "eda" not in st.session_state or st.session_state["eda_key"] != key:
        report = run_eda(df, plots_dir=PLOTS_DIR)
        st.session_state["eda"] = report
        st.session_state["eda_key"] = key
    return st.session_state["eda"]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_single_prediction() -> None:
    st.subheader("🔮 Single House Prediction")
    st.caption("Fill in the most impactful attributes. Everything else is imputed automatically.")

    with st.form("house_form"):
        c1, c2, c3 = st.columns(3)
        overall_qual = c1.slider("Overall quality", 1, 10, 6)
        overall_cond = c2.slider("Overall condition", 1, 10, 6)
        gr_liv_area = c3.number_input("Above-grade living area (sq ft)", 200, 10000, 1600)

        c4, c5, c6 = st.columns(3)
        total_bsmt = c4.number_input("Basement area (sq ft)", 0, 6000, 900)
        year_built = c5.number_input("Year built", 1800, 2025, 2003)
        bedrooms = c6.number_input("Bedrooms above grade", 0, 10, 3)

        c7, c8, c9 = st.columns(3)
        full_bath = c7.number_input("Full bathrooms", 0, 10, 2)
        half_bath = c8.number_input("Half bathrooms", 0, 10, 1)
        lot_area = c9.number_input("Lot area (sq ft)", 1000, 200000, 9000)

        c10, c11, c12 = st.columns(3)
        neighborhood = c10.text_input("Neighborhood", "NAmes")
        exter_qual = c11.selectbox("Exterior quality", ["Ex", "Gd", "TA", "Fa", "Po"])
        kitchen_qual = c12.selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa", "Po"])

        c13, c14, c15 = st.columns(3)
        fireplace_qu = c13.selectbox("Fireplace quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"])
        central_air = c14.selectbox("Central air", ["Y", "N"])
        garage_cars = c15.number_input("Garage capacity (cars)", 0, 6, 2)

        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        features = {
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt,
            "YearBuilt": year_built,
            "BedroomAbvGr": bedrooms,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "LotArea": lot_area,
            "Neighborhood": neighborhood,
            "ExterQual": exter_qual,
            "KitchenQual": kitchen_qual,
            "FireplaceQu": fireplace_qu,
            "CentralAir": central_air,
            "GarageCars": garage_cars,
        }
        service = get_service()
        result = service.predict_single(features)

        price = result["predicted_price"]
        st.metric("Predicted Sale Price", f"${price:,.0f}")

        exp = result.get("explanation", {})
        if exp and "explanation" in exp:
            st.markdown("#### How the model decided")
            df_exp = pd.DataFrame(exp["explanation"])
            st.bar_chart(
                df_exp.set_index("feature")["value"].head(10).sort_values(),
                horizontal=True,
            )
            st.caption(
                f"Base value (log-price): {exp.get('base_value', 0):.2f}. "
                "Positive bars push the price up; negative push it down."
            )
            waterfall = PLOTS_DIR / "shap_waterfall.png"
            if waterfall.exists():
                st.image(str(waterfall), caption="SHAP waterfall (last explained house)")


def page_batch_prediction() -> None:
    st.subheader("📁 Batch Prediction (CSV Upload)")
    st.caption(
        "Upload a CSV with house attributes. A prediction column is appended "
        "and you can download the results."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        if st.button("Run batch prediction"):
            service = get_service()
            predictions = service.predict_batch(df)
            st.success(f"Predicted {len(predictions)} houses")
            st.dataframe(predictions)

            buf = io.BytesIO()
            predictions.to_csv(buf, index=False)
            st.download_button(
                "Download predictions (CSV)",
                data=buf.getvalue().decode(),
                file_name="predictions.csv",
                mime="text/csv",
            )


def page_insights() -> None:
    st.subheader("📊 Data Insights")
    try:
        df = load_raw_data()
    except Exception as exc:
        st.warning(f"Raw data not available: {exc}")
        return

    if st.button("Run / refresh EDA"):
        run_eda(df, plots_dir=PLOTS_DIR)
        st.success("EDA refreshed")

    st.caption("Auto-generated plots from the EDA pipeline.")
    plots = [
        ("target_distribution.png", "Target distribution"),
        ("missing_values.png", "Missing values"),
        ("correlation_heatmap.png", "Correlation heatmap"),
        ("price_by_quality.png", "Price by quality"),
        ("price_vs_area.png", "Price vs living area"),
        ("price_by_neighborhood.png", "Price by neighborhood"),
    ]
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for col, (fname, title) in zip(cols, plots[i:i + 2], strict=False):
            path = PLOTS_DIR / fname
            if path.exists():
                col.image(str(path), caption=title, use_container_width=True)


def page_models() -> None:
    st.subheader("🤖 Model Comparison & Performance")
    leaderboard_path = settings.reports_dir / "model_leaderboard.csv"
    if leaderboard_path.exists():
        lb = pd.read_csv(leaderboard_path)
        st.dataframe(lb, use_container_width=True)
    else:
        st.info("No leaderboard found. Run `python scripts/run_training.py`.")

    comparison = PLOTS_DIR / "model_comparison.png"
    if comparison.exists():
        st.image(str(comparison), caption="Cross-validated model comparison",
                 use_container_width=True)

    diag = PLOTS_DIR / "holdout_diagnostics.png"
    if diag.exists():
        st.image(str(diag), caption="Hold-out diagnostics", use_container_width=True)

    summary_path = settings.reports_dir / "training_summary.json"
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.json(summary)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

PAGES = {
    "🔮 Single Prediction": page_single_prediction,
    "📁 Batch Prediction": page_batch_prediction,
    "📊 Data Insights": page_insights,
    "🤖 Model Comparison": page_models,
}

with st.sidebar:
    st.title("🏠 House Price")
    st.caption("Production ML platform")
    page = st.radio("Navigation", list(PAGES))

    st.divider()
    try:
        service = get_service()
        st.write("**Active model:**", service.metadata.get("model_name", "best_model"))
        st.write("**Source:**", service.metadata.get("source", "local"))
    except Exception:
        st.error("No trained model found. Run `python scripts/run_training.py`.")

    st.caption(f"API v{__version__}")

st.title("House Price Prediction Platform")
PAGES[page]()

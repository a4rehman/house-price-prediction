"""Exploratory Data Analysis.

Produces plots and a JSON summary of the dataset so insights are reproducible
and can be rendered on the dashboard without re-running analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import settings
from ..logging_config import get_logger
from ..utils import ensure_serialisable, save_plot

logger = get_logger(__name__)

PLOT_STYLE = "whitegrid"
PALETTE = "viridis"


@dataclass
class EdaReport:
    """Container holding paths + summary dict for an EDA run."""

    plots_dir: Path = settings.plots_dir
    reports_dir: Path = settings.reports_dir
    summary: dict = field(default_factory=dict)
    plot_paths: dict = field(default_factory=dict)

    def save_summary(self) -> Path:
        path = self.reports_dir / "eda_summary.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ensure_serialisable(self.summary), fh, indent=2)
        return path

    def save_html(self) -> Path:
        rows = "".join(
            f"<tr><td>{k}</td><td><img src='../plots/{Path(v).name}' style='max-width:100%;'/>"
            f"</td></tr>"
            for k, v in self.plot_paths.items()
        )
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
        <title>House Prices — EDA Report</title>
        <style>body{{font-family:sans-serif;margin:2rem;}} h1{{color:#2c3e50;}}
        .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
        tr{{vertical-align:top;}}</style></head><body>
        <h1>Exploratory Data Analysis — Ames House Prices</h1>
        <table class="grid">{rows}</table></body></html>"""
        path = self.reports_dir / "eda_report.html"
        path.write_text(html, encoding="utf-8")
        return path


def _setup_style() -> None:
    sns.set_theme(style=PLOT_STYLE, palette=PALETTE)


def run_eda(
    df: pd.DataFrame,
    target_col: str = "SalePrice",
    plots_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> EdaReport:
    """Execute the full EDA and persist plots + summary."""
    _setup_style()
    plots_dir = Path(plots_dir or settings.plots_dir)
    reports_dir = Path(reports_dir or settings.reports_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = EdaReport(plots_dir=plots_dir, reports_dir=reports_dir)
    target = df[target_col]

    report.summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report.summary["target"] = {
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "skew": float(target.skew()),
    }

    # ---- 1. Target distribution -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(target, kde=True, ax=axes[0], color="#2c3e50")
    axes[0].set_title("SalePrice distribution")
    sns.histplot(np.log1p(target), kde=True, ax=axes[1], color="#16a085")
    axes[1].set_title("log1p(SalePrice) distribution")
    fig.tight_layout()
    report.plot_paths["target_distribution"] = save_plot(
        fig, plots_dir / "target_distribution.png")

    # ---- 2. Missing values ------------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    report.summary["missing_values"] = {
        str(k): int(v) for k, v in missing.to_dict().items()
    }
    report.summary["missing_pct"] = {
        str(k): round(float(100 * v / len(df)), 2) for k, v in missing.to_dict().items()
    }
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        missing.plot(kind="barh", ax=ax, color="#e74c3c")
        ax.set_title("Missing values by column")
        ax.set_xlabel("Count")
        fig.tight_layout()
        report.plot_paths["missing_values"] = save_plot(
            fig, plots_dir / "missing_values.png")

    # ---- 3. Correlation heatmap -------------------------------------------
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    top_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
    report.summary["top_correlations"] = {
        str(k): round(float(v), 4) for k, v in top_corr.head(20).items()
    }
    top_cols = [target_col] + list(top_corr.head(15).index)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(numeric[top_cols].corr(), annot=True, fmt=".2f", cmap="viridis",
                linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap — top features vs SalePrice")
    fig.tight_layout()
    report.plot_paths["correlation_heatmap"] = save_plot(
        fig, plots_dir / "correlation_heatmap.png")

    # ---- 4. Price vs OverallQual ------------------------------------------
    if "OverallQual" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df, x="OverallQual", y=target_col, ax=ax, palette="viridis")
        ax.set_title("SalePrice by Overall Quality")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        report.plot_paths["price_by_quality"] = save_plot(
            fig, plots_dir / "price_by_quality.png")

    # ---- 5. Price vs GrLivArea --------------------------------------------
    if "GrLivArea" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=df, x="GrLivArea", y=target_col, hue="OverallQual",
                        palette="viridis", ax=ax, alpha=0.6)
        ax.set_title("SalePrice vs Above-Grade Living Area")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        report.plot_paths["price_vs_area"] = save_plot(
            fig, plots_dir / "price_vs_area.png")

    # ---- 6. Average price by Neighborhood ---------------------------------
    if "Neighborhood" in df.columns:
        nb = df.groupby("Neighborhood")[target_col].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        nb.plot(kind="bar", ax=ax, color="#8e44ad")
        ax.set_title("Median SalePrice by Neighborhood")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        report.plot_paths["price_by_neighborhood"] = save_plot(
            fig, plots_dir / "price_by_neighborhood.png")

    # ---- 7. Feature-type summary ------------------------------------------
    dtypes = df.dtypes.astype(str).value_counts().to_dict()
    report.summary["dtype_counts"] = {str(k): int(v) for k, v in dtypes.items()}
    report.summary["n_numeric"] = int(numeric.shape[1])
    report.summary["n_categorical"] = int(
        df.select_dtypes(include=["object"]).shape[1])

    plt.close("all")
    report.save_summary()
    report.save_html()
    logger.info(
        "EDA complete — %d plots, summary at %s",
        len(report.plot_paths), report.summary.get("shape"),
    )
    return report

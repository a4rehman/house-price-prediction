"""Tests for the EDA report generator."""

from __future__ import annotations

from src.data.eda import run_eda


def test_run_eda_produces_summary_and_plots(ames_synthetic, tmp_path):
    plots = tmp_path / "plots"
    reports = tmp_path / "reports"
    report = run_eda(ames_synthetic, plots_dir=plots, reports_dir=reports)

    assert "shape" in report.summary
    assert report.summary["shape"]["rows"] == len(ames_synthetic)
    assert "top_correlations" in report.summary
    assert len(report.plot_paths) >= 4

    summary_path = report.save_summary()
    assert summary_path.exists()
    html_path = report.save_html()
    assert html_path.exists()

    for path in report.plot_paths.values():
        assert path.exists()

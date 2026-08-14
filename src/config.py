"""Central configuration for the Customer Churn Intelligence platform.

All settings can be overridden through environment variables or a `.env`
file. Values are resolved at import time and exposed as a single `settings`
singleton used across the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    plots_dir: Path = PROJECT_ROOT / "artifacts" / "plots"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    mlflow_artifacts_dir: Path = PROJECT_ROOT / "artifacts" / "mlflow"
    mlruns_dir: Path = PROJECT_ROOT / "mlruns"

    # --- Data source -------------------------------------------------------
    data_url: str = ""
    raw_data_filename: str = "customer_churn.csv"
    target_column: str = "Churn"
    id_column: str = "customer_id"

    # --- Data quality ------------------------------------------------------
    na_values: list[str] = ["?", "NA", "N/A", "null"]
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    cap_extreme_lot_area: int = 100_000
    drop_extreme_grlivarea_price: tuple[float, float] = (4_000, 300_000)

    # --- Modelling ---------------------------------------------------------
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    n_trials: int = 30
    use_log_target: bool = True

    # --- Model registry ----------------------------------------------------
    registered_model_name: str = "customer_churn_predictor"
    default_model_version: str = "latest"
    model_uri: str = ""  # e.g. mlflow://models:/house_price_predictor/Production

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_file: str = "logs/churn.log"

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_batch_size: int = 10_000

    # --- Dashboard ---------------------------------------------------------
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501

    def ensure_dirs(self) -> None:
        """Create all runtime directories if they do not exist."""
        for path in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.plots_dir,
            self.reports_dir,
            self.mlflow_artifacts_dir,
            self.mlruns_dir,
            self.project_root / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

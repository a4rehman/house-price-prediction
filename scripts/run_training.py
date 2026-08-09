"""Train, compare, tune, and register the house price model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_config import setup_logging
from src.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow registration")
    args = parser.parse_args()

    setup_logging()
    summary = run_training(
        tune=not args.no_tune,
        n_trials=args.trials,
        force_download=args.force_download,
        register_mlflow=not args.no_mlflow,
    )
    print("\n=== Training summary ===")
    print(f"Best model:  {summary['best_model']}")
    print(f"CV RMSE(log): {summary['cv_rmse_log']:.4f}")
    print(f"Hold-out:    {summary['holdout_metrics']}")
    print(f"Artifacts:   {summary['artifacts_dir']}")


if __name__ == "__main__":
    main()

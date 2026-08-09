"""Run EDA on the raw dataset and write plots + a JSON summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.data.eda import run_eda
from src.data.loader import download_raw_data, load_raw_data
from src.data.preprocessing import remove_outliers
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA and save artifacts")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-outliers", action="store_true")
    args = parser.parse_args()

    setup_logging()
    download_raw_data(force=args.force_download)
    df = load_raw_data()
    if not args.skip_outliers:
        df = remove_outliers(df, target_col=settings.target_column)
    report = run_eda(df)
    logger.info("EDA report: %d plots, summary at %s",
                len(report.plot_paths), report.save_summary())


if __name__ == "__main__":
    main()

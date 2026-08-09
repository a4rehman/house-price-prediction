"""Batch prediction from a CSV file using the trained local model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_config import get_logger, setup_logging
from src.models.inference import PredictionService

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict from CSV")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    setup_logging()
    df = pd.read_csv(args.input)
    service = PredictionService().load()
    predictions = service.predict_batch(df)

    output = args.output or args.input.rsplit(".", 1)[0] + "_predictions.csv"
    predictions.to_csv(output, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), output)


if __name__ == "__main__":
    main()

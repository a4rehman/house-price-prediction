"""Download the Ames house prices dataset into data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import download_raw_data
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw dataset")
    parser.add_argument("--url", default=None, help="Override download URL")
    parser.add_argument("--force", action="store_true", help="Re-download if present")
    args = parser.parse_args()

    setup_logging()
    path = download_raw_data(url=args.url, force=args.force)
    logger.info("Data ready at %s", path)


if __name__ == "__main__":
    main()

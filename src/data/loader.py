"""Data ingestion and download utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


def download_raw_data(
    url: str | None = None,
    destination: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download the Ames house prices CSV from the configured URL.

    Args:
        url: Source URL (defaults to ``settings.data_url``).
        destination: Output path (defaults to ``data/raw/house_prices.csv``).
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded file.
    """
    url = url or settings.data_url
    destination = Path(destination or settings.data_raw_dir / settings.raw_data_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        logger.info("Raw data already present: %s", destination)
        return destination

    logger.info("Downloading data from %s", url)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

    logger.info("Saved raw data to %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Ames dataset using the configured NA markers."""
    path = Path(path or settings.data_raw_dir / settings.raw_data_filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, na_values=settings.na_values)
    logger.info("Loaded raw data: %d rows x %d columns", *df.shape)
    return df


def load_processed_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load a previously saved processed dataset (CSV)."""
    path = Path(path or settings.data_processed_dir / "dataset.csv")
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}.")
    return pd.read_csv(path)

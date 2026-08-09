"""Start the Streamlit prediction dashboard (dev convenience wrapper)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard")
    parser.add_argument("--host", default=settings.dashboard_host)
    parser.add_argument("--port", type=int, default=settings.dashboard_port)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--server.headless", "true",
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

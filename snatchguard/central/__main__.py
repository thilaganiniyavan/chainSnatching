"""Run the central server + dashboard.

    python -m snatchguard.central                  # http://127.0.0.1:8000
    python -m snatchguard.central --host 0.0.0.0   # reachable from edge boxes on the network
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from snatchguard.central.app import create_app
from snatchguard.core.config import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="SnatchGuard central server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(REPO_ROOT / "runtime" / "central"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [central] %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(args.data), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Run an edge node.

    python -m snatchguard.edge                                   # configs/edge.yaml
    python -m snatchguard.edge --source data/samples/clip.mp4    # one camera from a file (plays in real time)
    python -m snatchguard.edge --source clip.mp4 --night ir      # ...turned into IR night footage
    python -m snatchguard.edge --source rtsp://user:pw@cam/stream
    python -m snatchguard.edge --source synthetic:snatch         # scripted scene, no models needed
"""

from __future__ import annotations

import argparse
import logging

from snatchguard.core.config import REPO_ROOT, load_yaml
from snatchguard.edge.node import EdgeNode


def main() -> None:
    ap = argparse.ArgumentParser(description="SnatchGuard edge node")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--source", help="run a single camera from this source instead of the config's camera list")
    ap.add_argument("--camera-id", default="cam-01")
    ap.add_argument("--name", help="camera display name")
    ap.add_argument("--night", choices=["none", "ir", "lowlight"], default="none",
                    help="simulate night by transforming the source video")
    ap.add_argument("--lighting", choices=["auto", "day", "lowlight", "ir"], default="auto")
    ap.add_argument("--loop", action="store_true", help="loop a video file forever")
    ap.add_argument("--central", help="central server URL (overrides config)")
    ap.add_argument("--until-done", action="store_true", help="exit when all sources ended and alerts are delivered")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [node] %(levelname)s %(name)s: %(message)s")
    cfg = load_yaml(args.config)
    if args.central:
        cfg.setdefault("central", {})["url"] = args.central
    if args.source:
        cfg["cameras"] = [{
            "id": args.camera_id, "name": args.name or args.source.split("/")[-1],
            "source": args.source, "loop": args.loop, "realtime": True,
            "lighting": args.lighting, "simulate_night": args.night,
        }]
    EdgeNode(cfg, log_level=level).run(until_done=args.until_done)


if __name__ == "__main__":
    main()

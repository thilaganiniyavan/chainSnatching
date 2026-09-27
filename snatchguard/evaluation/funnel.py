"""Where does each clip get lost? A stage-by-stage funnel through the live pipeline.

    detected bike -> detected person -> person established on foot -> pair opened -> contact -> alert

    python -m snatchguard.evaluation.funnel --only snatch_bike
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from snatchguard.core.config import REPO_ROOT, load_yaml
from snatchguard.core.types import Level, ObjClass
from snatchguard.evaluation.analyse import build_pipeline
from snatchguard.evaluation.benchmark import DATA, load_labels

STAGES = ["bike", "person", "established", "pair", "contact", "alert"]


def funnel(path: str, cfg: dict, fps: float = 8.0) -> dict:
    pipe = build_pipeline(cfg)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    reached = dict.fromkeys(STAGES, False)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if (idx - 1) % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        if img.shape[1] > 1280:
            img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
        res = pipe.process(img, (idx - 1) / src_fps)
        eng = pipe.engine
        reached["bike"] |= any(d.cls.is_vehicle for d in res.detections)
        reached["person"] |= any(d.cls is ObjClass.PERSON for d in res.detections)
        reached["established"] |= bool(eng.scene.pedestrians())
        reached["pair"] |= bool(eng.pairs)
        reached["contact"] |= any(p.phase != "WATCH" for p in eng.pairs.values())
        reached["alert"] |= any(d.level is not Level.DISMISSED for d in res.decisions)
    reached["alert"] |= any(d.level is not Level.DISMISSED for d in pipe.engine.flush())
    return reached


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--only", default="snatch_bike")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    totals = dict.fromkeys(STAGES, 0)
    rows = [r for r in load_labels(Path(args.labels)) if r["category"] == args.only]
    print(f"{'clip':28s} " + " ".join(f"{s:>11s}" for s in STAGES))
    for row in rows:
        path = DATA / row["video"]
        if not path.exists():
            continue
        r = funnel(str(path), cfg)
        for s in STAGES:
            totals[s] += r[s]
        print(f"{row['video']:28s} " + " ".join(f"{'yes' if r[s] else '-':>11s}" for s in STAGES), flush=True)
    print(f"{'TOTAL of ' + str(len(rows)):28s} " + " ".join(f"{totals[s]:>11d}" for s in STAGES))


if __name__ == "__main__":
    main()

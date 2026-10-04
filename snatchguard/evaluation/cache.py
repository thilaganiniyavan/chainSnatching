"""Perception cache: run the expensive models once per video, replay the rules as often as needed.

Detection, tracking and pose dominate the cost (a GPU-hour for the whole Snatch 1.0 set); the rule
engine replays a cached video in seconds. That makes threshold tuning and cross-validation over
hours of footage practical.

A cache is tied to the perception setup (models, input size, tracker, sampling rate) through a short
tag, so changing models never mixes old and new results.

    python -m snatchguard.evaluation.cache --config configs/edge.gpu.yaml          # all labelled videos
    python -m snatchguard.evaluation.cache --only snatch_bike --config ...
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from snatchguard.core.config import REPO_ROOT, RuleConfig, load_yaml
from snatchguard.core.types import Detection, FrameObservation, Lighting, ObjClass, Pose
from snatchguard.rules.engine import RuleEngine

CACHE_DIR = REPO_ROOT / "runtime" / "cache"
CACHE_VERSION = 1


@dataclass
class CachedVideo:
    video: str
    tag: str
    src_fps: float
    sample_fps: float
    duration_s: float
    frames: list[FrameObservation] = field(default_factory=list)
    build_fps: float = 0.0


def perception_tag(edge_cfg: dict, fps: float, night: str | None = None) -> str:
    m = edge_cfg["models"]
    tracker = Path(edge_cfg.get("tracker", "configs/tracker.yaml"))
    tracker = tracker if tracker.is_absolute() else REPO_ROOT / tracker
    key = json.dumps({
        "v": CACHE_VERSION, "det": str(m["detector"]), "pose": str(m.get("pose")), "imgsz": m.get("imgsz"),
        "pose_imgsz": m.get("pose_imgsz"), "pose_mode": m.get("pose_mode"), "conf": m.get("conf"),
        "fps": fps, "tracker": tracker.read_text() if tracker.exists() else "",
        **({"night": night} if night and night != "none" else {}),
    }, sort_keys=True)
    suffix = f"-{night}" if night and night != "none" else ""
    return (Path(str(m["detector"])).name.replace(".", "_")[:24] + "-" + hashlib.md5(key.encode()).hexdigest()[:8]
            + suffix)


def cache_path(video: str, tag: str) -> Path:
    stem = Path(video).parent.name.replace(" ", "_") + "__" + Path(video).stem
    return CACHE_DIR / tag / f"{stem}.pkl.gz"


# ---- (de)serialisation: plain tuples keep the files small and version-proof ----------------------

def _pack(frames: list[FrameObservation]) -> list:
    out = []
    for f in frames:
        dets = [(d.track_id, d.cls.value, d.box.astype(np.float32), round(d.conf, 3)) for d in f.detections]
        poses = {tid: (p.xy.astype(np.float32), p.conf.astype(np.float16)) for tid, p in f.poses.items()}
        out.append((f.t, f.frame_size, f.lighting.value, dets, poses))
    return out


def _unpack(rows: list) -> list[FrameObservation]:
    frames = []
    for t, size, light, dets, poses in rows:
        frames.append(FrameObservation(
            t=t, frame_size=tuple(size), lighting=Lighting(light),
            detections=[Detection(tid, ObjClass(c), np.asarray(b, dtype=float), conf) for tid, c, b, conf in dets],
            poses={tid: Pose(np.asarray(xy, dtype=float), np.asarray(cf, dtype=float)) for tid, (xy, cf) in poses.items()},
        ))
    return frames


def save(cv: CachedVideo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=5) as f:
        pickle.dump({"video": cv.video, "tag": cv.tag, "src_fps": cv.src_fps, "sample_fps": cv.sample_fps,
                     "duration_s": cv.duration_s, "build_fps": cv.build_fps, "frames": _pack(cv.frames)}, f)
    tmp.replace(path)


def load(path: Path) -> CachedVideo:
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    return CachedVideo(d["video"], d["tag"], d["src_fps"], d["sample_fps"], d["duration_s"],
                       _unpack(d["frames"]), d.get("build_fps", 0.0))


# ---- building -------------------------------------------------------------------------------------

def build(video: str, edge_cfg: dict, fps: float = 8.0, night: str | None = None,
          progress: bool = False, max_seconds: float | None = None) -> CachedVideo:
    """Run the live perception on every sampled frame and record what the rules would see."""
    from snatchguard.evaluation.analyse import build_pipeline
    from snatchguard.sim.night import make_transform

    pipe = build_pipeline(edge_cfg)
    transform = make_transform(night)
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    cv = CachedVideo(video, perception_tag(edge_cfg, fps, night), src_fps, src_fps / step, 0.0)
    idx, t0 = 0, time.perf_counter()
    while True:
        if not cap.grab():
            break
        idx += 1
        if (idx - 1) % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        t = (idx - 1) / src_fps
        if max_seconds is not None and t > max_seconds:
            break
        if img.shape[1] > 1280:
            img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])), interpolation=cv2.INTER_AREA)
        if transform is not None:
            img = transform(img)
        res = pipe.process(img, t)
        h, w = img.shape[:2]
        cv.frames.append(FrameObservation(t, (w, h), res.detections, res.poses or {}, res.mode))
        cv.duration_s = t
        if progress and len(cv.frames) % 400 == 0:
            print(f"  {Path(video).name}: {t:7.1f}s, {len(cv.frames) / (time.perf_counter() - t0):.1f} fps", flush=True)
    cv.build_fps = len(cv.frames) / max(time.perf_counter() - t0, 1e-6)
    return cv


def get(video: str, edge_cfg: dict, fps: float = 8.0, rebuild: bool = False, progress: bool = False) -> CachedVideo:
    path = cache_path(video, perception_tag(edge_cfg, fps))
    if path.exists() and not rebuild:
        return load(path)
    cv = build(video, edge_cfg, fps, progress=progress)
    save(cv, path)
    return cv


# ---- replay ---------------------------------------------------------------------------------------

def replay(cv: CachedVideo, rules: RuleConfig) -> list[dict]:
    """Rule decisions for a cached video (same engine as live, plus the end-of-stream flush)."""
    engine = RuleEngine(rules)
    out = []
    for f in cv.frames:
        for d in engine.step(f):
            p = d.to_payload()
            p["video_time_s"] = round(f.t, 2)
            out.append(p)
    for d in engine.flush():
        p = d.to_payload()
        p["video_time_s"] = round(cv.duration_s, 2)
        p["flushed_at_end"] = True
        out.append(p)
    return out


def main() -> None:
    from snatchguard.evaluation.benchmark import DATA, load_labels

    ap = argparse.ArgumentParser(description="Build perception caches for labelled videos")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--only", help="category (snatch_bike, snatch_foot, normal)")
    ap.add_argument("--folds", default="A,B", help="folds to cache (default: the evaluation folds)")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-seconds", type=float, help="cap each video (smoke tests, night evaluation)")
    ap.add_argument("--night", choices=["none", "ir", "lowlight"], default="none",
                    help="turn the footage into simulated night first (night evaluation)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    folds = set(args.folds.split(","))
    rows = [r for r in load_labels(Path(args.labels))
            if r.get("fold") in folds and (not args.only or r["category"] == args.only)]
    tag = perception_tag(cfg, args.fps, args.night)
    print(f"perception tag {tag}: {len(rows)} videos -> {CACHE_DIR / tag}")
    for r in rows:
        path = Path(args.data) / r["video"]
        if not path.exists():
            print(f"  missing {path} (download the dataset first)")
            continue
        out = cache_path(str(path), tag)
        if out.exists() and not args.rebuild:
            print(f"  cached  {r['video']}")
            continue
        t0 = time.time()
        cv = build(str(path), cfg, args.fps, night=args.night, progress=True, max_seconds=args.max_seconds)
        save(cv, out)
        print(f"  built   {r['video']}: {len(cv.frames)} frames, {cv.duration_s:.0f}s video "
              f"in {time.time() - t0:.0f}s ({cv.build_fps:.1f} fps)", flush=True)


if __name__ == "__main__":
    main()

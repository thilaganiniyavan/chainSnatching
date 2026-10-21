"""Feature cache: run the frozen backbone once per video; training and evaluation then work from the cache.

    python -m snatchguard.learned.extract --labels data/labels/snatch1.0.csv --data "data/samples/snatch1.0/Chain Snatching Videos"
    python -m snatchguard.learned.extract --labels ~/snatch_web_dataset/labels.csv --data ~/snatch_web_dataset/clips
    ... --normal-max-seconds 600     # only the first 10 min of each normal video (slow CPUs)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from snatchguard.core.config import REPO_ROOT
from snatchguard.learned.features import Backbone, FeatureSpec, FrameFeaturizer
from snatchguard.learned.scorer import iter_video

FEATURES_DIR = REPO_ROOT / "runtime" / "features"
IN_SCOPE = ("snatch_bike", "snatch_foot")
DEFAULT_LABELS = REPO_ROOT / "data" / "labels" / "snatch1.0.csv"
DEFAULT_DATA = REPO_ROOT / "data" / "samples" / "snatch1.0" / "Chain Snatching Videos"


def video_stem(video: str | Path) -> str:
    """Same naming as snatchguard.evaluation.cache.cache_path: '<parent dir>__<file stem>'."""
    p = Path(video)
    return p.parent.name.replace(" ", "_") + "__" + p.stem


def feature_path(video: str | Path, spec: FeatureSpec, root: Path | None = None) -> Path:
    return (root or FEATURES_DIR) / spec.tag / f"{video_stem(video)}.npz"


def video_duration(path: str | Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps, fps


def extract_video(path: str | Path, spec: FeatureSpec, backbone: Backbone, max_seconds: float | None = None,
                  progress: bool = False) -> dict:
    feat = FrameFeaturizer(spec, backbone)
    ts, embs, mots = [], [], []
    t0 = time.perf_counter()
    for t, img in iter_video(path, spec.sample_fps, max_seconds):
        e, m = feat(img)
        ts.append(t)
        embs.append(e.astype(np.float16))
        mots.append(m)
        if progress and len(ts) % 400 == 0:
            print(f"    {Path(path).name}: {t:7.1f}s  {len(ts) / (time.perf_counter() - t0):.2f} samples/s", flush=True)
    full, src_fps = video_duration(path)
    dur = min(full, max_seconds) if max_seconds else full
    if ts:
        dur = max(dur, ts[-1])
    dim = backbone.dim
    return {"t": np.array(ts, dtype=np.float64),
            "emb": np.stack(embs) if embs else np.zeros((0, spec.n_views, dim), np.float16),
            "motion": np.stack(mots) if mots else np.zeros((0, spec.n_views, 3), np.float32),
            "duration_s": np.float64(dur), "src_fps": np.float64(src_fps),
            "max_seconds": np.float64(max_seconds if max_seconds else -1),
            "build_samples_per_s": np.float64(len(ts) / max(time.perf_counter() - t0, 1e-6)),
            "spec_tag": np.array(spec.tag)}


def save_features(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(path)


def load_features(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_meta(path: Path) -> dict:
    """Everything except the embeddings (cheap)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "emb"}


def is_complete(path: Path, max_seconds: float | None) -> bool:
    """A cache made with a cap is enough only for runs with the same or a smaller cap."""
    if not path.exists():
        return False
    stored = float(load_meta(path)["max_seconds"])
    if stored < 0:
        return True
    return max_seconds is not None and stored >= max_seconds


def select_rows(labels: Path, folds: set[str]) -> list[dict]:
    with open(labels, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["category"] in IN_SCOPE or r["category"] == "normal") and r.get("fold") in folds]
    return rows


def run(labels: Path, data: Path, folds: set[str], spec: FeatureSpec, device: str = "cpu", threads: int = 2,
        normal_max_seconds: float | None = None, rebuild: bool = False, root: Path | None = None,
        only: set[str] | None = None) -> list[Path]:
    rows = [r for r in select_rows(labels, folds) if not only or r["video"] in only]
    todo = []
    for r in rows:
        path = data / r["video"]
        if not path.exists():
            print(f"  missing {path}")
            continue
        cap = normal_max_seconds if r["category"] == "normal" else None
        out = feature_path(r["video"], spec, root)
        if not rebuild and is_complete(out, cap):
            continue
        todo.append((video_duration(path)[0] if cap is None else min(cap, video_duration(path)[0]), r, path, out, cap))
    todo.sort(key=lambda x: (x[1]["category"] == "normal", x[0]))
    print(f"features {spec.tag}: {len(rows)} videos, {len(todo)} to extract "
          f"({sum(x[0] for x in todo) / 60:.1f} min of video) -> {(root or FEATURES_DIR) / spec.tag}", flush=True)
    if not todo:
        return [feature_path(r["video"], spec, root) for r in rows]
    cv2.setNumThreads(threads)
    backbone = Backbone(spec.backbone, device, threads)
    for dur, r, path, out, cap in todo:
        t0 = time.time()
        d = extract_video(path, spec, backbone, cap, progress=True)
        save_features(d, out)
        print(f"  {r['video']}: {len(d['t'])} samples of {float(d['duration_s']):.0f}s in {time.time() - t0:.0f}s "
              f"({float(d['build_samples_per_s']):.2f} samples/s)", flush=True)
    return [feature_path(r["video"], spec, root) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract learned-scorer features for labelled videos")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--folds", default="A,B")
    ap.add_argument("--device", default="cpu", help="cpu or a CUDA index such as 0")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--normal-max-seconds", type=float, default=None,
                    help="only the first N seconds of each normal video (snatch clips are always whole)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", action="append", help="just this video (labels.csv `video` field); repeatable")
    ap.add_argument("--list", action="store_true", help="print the videos that still need extracting and exit")
    args = ap.parse_args()
    labels, data, folds = args.labels.expanduser(), args.data.expanduser(), set(args.folds.split(","))
    if args.list:
        for r in select_rows(labels, folds):
            cap = args.normal_max_seconds if r["category"] == "normal" else None
            if (data / r["video"]).exists() and (args.rebuild or not is_complete(feature_path(r["video"], FeatureSpec()), cap)):
                print(r["video"])
        return
    run(labels, data, folds, FeatureSpec(), args.device, args.threads, args.normal_max_seconds, args.rebuild,
        only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()

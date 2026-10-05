"""Stage 3a: find the fixed-camera (CCTV) segments inside each downloaded video.

News uploads mix CCTV footage with anchors, graphics, phone footage and slow-motion replays.
For each video: detect shot cuts, keep shots where the camera does not move (global shift from
phase correlation stays near zero) and that last at least 3 s, and among repeated shots of the same
footage inside one video keep the shortest one (the real-time version, not the slow-motion replay).

    python -m snatchguard.webdata.segment --kind snatch
Output: ~/snatch_web_dataset/meta/segments_<kind>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from snatchguard.webdata.discover import DATASET
from snatchguard.webdata.fingerprint import compute, similarity

SAMPLE_FPS = 5.0
CUT_DIFF = 35.0          # mean abs difference (0-255) between samples 0.2 s apart = a cut
STATIC_SHIFT_PX = 1.0    # median global shift per sample (at 160 px width) for a fixed camera
MIN_SEGMENT_S = 3.0


def analyse(path: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / SAMPLE_FPS))
    frames, times = [], []
    idx = 0
    while True:
        if not cap.grab():
            break
        idx += 1
        if (idx - 1) % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        frames.append(g.astype(np.float32))
        times.append((idx - 1) / fps)
    if len(frames) < 5:
        return []
    win = cv2.createHanningWindow((160, 90), cv2.CV_32F)
    diffs, shifts = [0.0], [0.0]
    for a, b in zip(frames, frames[1:]):
        diffs.append(float(np.abs(a - b).mean()))
        (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), win)   # it windows its inputs in place
        shifts.append(float(np.hypot(dx, dy)))
    # shots
    bounds = [0] + [i for i, d in enumerate(diffs) if d >= CUT_DIFF] + [len(frames)]
    segments = []
    for s, e in zip(bounds, bounds[1:]):
        if e - s < 2:
            continue
        t0, t1 = times[s], times[e - 1]
        if t1 - t0 < MIN_SEGMENT_S:
            continue
        sh = np.array(shifts[s + 1:e]) if e - s > 1 else np.array([0.0])
        static = float(np.median(sh)) <= STATIC_SHIFT_PX and float((sh > 3.0).mean()) <= 0.15
        segments.append({"start": round(t0, 2), "end": round(t1, 2), "static": static,
                         "median_shift": round(float(np.median(sh)), 2)})
    return segments


def drop_replays(path: Path, segs: list[dict]) -> list[dict]:
    """Same footage shown twice (e.g. real time, then slow motion): keep the shortest copy."""
    static = [s for s in segs if s["static"]]
    fps_ = {i: compute(path, max_seconds=s["end"] - s["start"], start_s=s["start"]) for i, s in enumerate(static)}
    keep = set(range(len(static)))
    for i in range(len(static)):
        for j in range(i + 1, len(static)):
            if i in keep and j in keep and fps_[i] and fps_[j] and similarity(fps_[i], fps_[j])["same_footage"]:
                di, dj = static[i]["end"] - static[i]["start"], static[j]["end"] - static[j]["start"]
                keep.discard(j if dj >= di else i)
    for i, s in enumerate(static):
        s["replay_of_other"] = i not in keep
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["snatch", "normal"], required=True)
    args = ap.parse_args()
    out = DATASET / "meta" / f"segments_{args.kind}.json"
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    vids = sorted((DATASET / "raw" / args.kind).glob("*.mp4"))
    for n, p in enumerate(vids, 1):
        if p.stem in done:
            continue
        try:
            segs = analyse(p)
            if args.kind == "snatch":
                segs = drop_replays(p, segs)
        except Exception as exc:  # noqa: BLE001 - a broken download must not stop the batch
            segs = [{"error": str(exc)}]
        done[p.stem] = segs
        good = [s for s in segs if s.get("static") and not s.get("replay_of_other")]
        print(f"[{n}/{len(vids)}] {p.stem}: {len(segs)} shots, {len(good)} fixed-camera "
              f"({sum(s['end'] - s['start'] for s in good):.0f} s)", flush=True)
        out.write_text(json.dumps(done, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

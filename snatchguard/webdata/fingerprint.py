"""Video fingerprints for de-duplication (no ML).

Per frame sampled at 2 fps (centre 80 % crop, grayscale):
  * appearance hash: 64-bit difference hash of the frame       -> same camera view
  * motion hash: 64-bit difference hash of |frame - frame 0.5 s earlier|, only when something moved
                                                                -> same actual footage
Two uploads of the same incident share motion hashes even with different logos, borders, resolution
or compression; two different recordings from the same fixed camera share appearance but not motion.

    python -m snatchguard.webdata.fingerprint --snatch1 ~/snatchguard/data/samples/snatch1.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SAMPLE_FPS = 2.0            # new videos
REFERENCE_FPS = 5.0         # the reference set is sampled densely so any frame has a match within 0.1 s
MOTION_MIN = 2.5            # mean abs difference (0-255) below this = nothing moved, motion hash unusable
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class Fingerprint:
    app: np.ndarray        # uint64 per sample
    motion: np.ndarray     # uint64 per sample (only samples with motion)
    times: np.ndarray      # seconds of each appearance sample
    duration: float
    width: int
    height: int


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def compute(path: str | Path, max_seconds: float | None = None, start_s: float = 0.0,
            sample_fps: float = SAMPLE_FPS) -> Fingerprint | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / sample_fps))
    app, mot, times = [], [], []
    prev = None
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
        t = start_s + (idx - 1) / fps
        if max_seconds and t - start_s > max_seconds:
            break
        H, W = img.shape[:2]
        crop = img[int(0.1 * H):int(0.9 * H), int(0.1 * W):int(0.9 * W)]
        gray = cv2.cvtColor(cv2.resize(crop, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        app.append(_dhash(gray))
        times.append(t)
        if prev is not None and len(times) >= 2:
            diff = cv2.absdiff(gray, prev)
            if float(diff.mean()) >= MOTION_MIN:
                mot.append(_dhash(diff))
        prev = gray
    return Fingerprint(np.array(app, dtype=np.uint64), np.array(mot, dtype=np.uint64), np.array(times),
                       times[-1] if times else 0.0, w, h)


def save(fp: Fingerprint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, app=fp.app, motion=fp.motion, times=fp.times,
                        meta=np.array([fp.duration, fp.width, fp.height]))


def load(path: Path) -> Fingerprint:
    d = np.load(path)
    dur, w, h = d["meta"]
    return Fingerprint(d["app"], d["motion"], d["times"], float(dur), int(w), int(h))


def _min_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every hash in a, the smallest Hamming distance to any hash in b."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), 64, dtype=np.uint8)
    out = np.empty(len(a), dtype=np.uint8)
    for i in range(0, len(a), 256):
        x = (a[i:i + 256, None] ^ b[None, :]).view(np.uint8).reshape(min(256, len(a) - i), len(b), 8)
        out[i:i + 256] = _POP[x].sum(axis=2).min(axis=1)
    return out


def match_fraction(a: np.ndarray, b: np.ndarray, tol: int) -> float:
    return float((_min_hamming(a, b) <= tol).mean()) if len(a) else 0.0


def similarity(x: Fingerprint, y: Fingerprint) -> dict:
    """Containment in both directions (a short clip inside a long news upload still matches)."""
    mot = max(match_fraction(x.motion, y.motion, 8), match_fraction(y.motion, x.motion, 8)) \
        if min(len(x.motion), len(y.motion)) >= 5 else 0.0
    app = max(match_fraction(x.app, y.app, 6), match_fraction(y.app, x.app, 6))
    same = (mot >= 0.25 and app >= 0.40) or (app >= 0.90 and mot >= 0.10)
    return {"motion": round(mot, 3), "appearance": round(app, 3),
            "same_footage": same,
            "possible": not same and app >= 0.60}     # same camera view or a static scene: check by eye


def main() -> None:
    from snatchguard.webdata.discover import DATASET
    ap = argparse.ArgumentParser(description="Fingerprint the Snatch 1.0 reference set")
    ap.add_argument("--snatch1", type=Path, default=Path.home() / "snatchguard" / "data" / "samples" / "snatch1.0")
    args = ap.parse_args()
    out = DATASET / "meta" / "fp_snatch1"
    vids = sorted(p for p in args.snatch1.rglob("*") if p.suffix.lower() in (".avi", ".mp4", ".mkv", ".mov"))
    for p in vids:
        dest = out / f"{p.parent.name.replace(' ', '_')}__{p.stem}.npz"
        if dest.exists():
            continue
        fp = compute(p, sample_fps=REFERENCE_FPS)
        if fp is None:
            print(f"unreadable {p.name}")
            continue
        save(fp, dest)
        print(f"{p.parent.name}/{p.name}: {len(fp.app)} samples, {len(fp.motion)} with motion", flush=True)


if __name__ == "__main__":
    main()

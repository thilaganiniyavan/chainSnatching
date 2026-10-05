"""Stage 3c: score how much *moving* graphics (news circles, arrows, spotlights) a segment has.

Broadcast annotations are drawn in pure, fully saturated colours (red, yellow, green, cyan, magenta)
that compressed CCTV of a real scene almost never reaches. Static graphics (channel logo, ticker,
DVR menu) occlude a fixed part of the frame and are tolerable; graphics that move or blink cover
the people and change from frame to frame, which no real camera does. So:

    graphic(t) = pixels with S >= 220 and V >= 200 at time t
    static     = pixels graphic in at least half of the samples (logos, tickers)
    moving(t)  = graphic(t) minus static, after a small opening to drop isolated pixels
    score      = 90th percentile over t of the moving fraction of the frame

    python -m snatchguard.webdata.overlay --kind snatch          # all ok/check segments
Output: ~/snatch_web_dataset/meta/overlay_<kind>.json  (key -> {"score", "static_frac"})
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from snatchguard.webdata.discover import DATASET

SAMPLE_FPS = 2.0
WIDTH = 320
OVERLAY_SCORE = 0.001     # above this: certainly annotated (moving pure-colour graphics or rotating ads); below it
                          # thin/white rings can still hide, so the human review stays the final check


def sample_frames(path: Path, start: float, end: float, fps: float, max_samples: int | None = None):
    """Yield (t, frame) every 1/fps seconds from start to end, decoding sequentially (one seek only)."""
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(vfps / fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    idx, n = 0, 0
    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end + 1e-3:
            break
        if idx % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            yield t, img
            n += 1
            if max_samples and n >= max_samples:
                break
        idx += 1
    cap.release()


def graphic_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] >= 220) & (hsv[..., 2] >= 200)


def score(path: Path, start: float, end: float) -> dict:
    masks = []
    for _, img in sample_frames(path, start, end, SAMPLE_FPS, max_samples=240):
        h = round(img.shape[0] * WIDTH / img.shape[1])
        masks.append(graphic_mask(cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA)))
    if len(masks) < 2:
        return {"score": None, "static_frac": None, "samples": len(masks)}
    stack = np.stack(masks)
    static = stack.mean(axis=0) >= 0.5
    kernel = np.ones((2, 2), np.uint8)
    moving = [cv2.morphologyEx((m & ~static).astype(np.uint8), cv2.MORPH_OPEN, kernel).mean() for m in stack]
    return {"score": round(float(np.percentile(moving, 90)), 5), "static_frac": round(float(static.mean()), 4),
            "samples": len(masks)}


def _unclear(kind: str) -> list[tuple[str, dict]]:
    from snatchguard.webdata.review import decisions
    return [(k, d) for k, d in decisions(kind).items() if d["decision"] == "unclear"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["snatch", "normal"], required=True)
    ap.add_argument("--keys", nargs="*", help="only these segment keys")
    ap.add_argument("--all", action="store_true", help="every ok/check segment, not only review candidates")
    args = ap.parse_args()
    from snatchguard.webdata.review import candidates
    dd = json.loads((DATASET / "meta" / f"dedup_{args.kind}.json").read_text(encoding="utf-8"))
    out = DATASET / "meta" / f"overlay_{args.kind}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if args.keys:
        keys = args.keys
    elif args.all:
        keys = [k for k, v in dd.items() if v["status"] in ("ok", "check")]
    else:   # unreviewed segments that pass the pre-screen and length rules, plus the 'unclear' ones
        keys = [k for k, _ in candidates(args.kind)] + [k for k, d in _unclear(args.kind)]
    for n, key in enumerate(keys, 1):
        if key in res:
            continue
        seg = dd[key]
        res[key] = score(DATASET / "raw" / args.kind / f"{key.split('#')[0]}.mp4", seg["start"], seg["end"])
        print(f"[{n}/{len(keys)}] {key}: {res[key]}", flush=True)
        if n % 20 == 0:
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

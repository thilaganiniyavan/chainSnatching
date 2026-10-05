"""Stage 3b: remove anything that repeats Snatch 1.0, and repeats within the new set.

For every fixed-camera segment (from segment.py), compute a fingerprint and compare it with
  * every Snatch 1.0 video (dense reference fingerprints from fingerprint.py)  -> dup_snatch1
  * every segment already accepted from another upload                         -> dup_web
Segments are processed best-first (higher resolution, longer), so of several uploads of the same
footage the best copy is kept. Borderline appearance matches are flagged for a visual check.

    python -m snatchguard.webdata.dedup --kind snatch
Output: ~/snatch_web_dataset/meta/dedup_<kind>.json  (segment key -> status, match, scores)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from snatchguard.webdata import fingerprint as F
from snatchguard.webdata.discover import DATASET


def load_reference() -> dict[str, F.Fingerprint]:
    """Snatch 1.0 fingerprints, keyed like 'Snatch_Theft__17_0' or 'Normal__s_2'."""
    return {p.stem: F.load(p) for p in sorted((DATASET / "meta" / "fp_snatch1").glob("*.npz"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["snatch", "normal"], required=True)
    ap.add_argument("--min-seconds", type=float, default=3.0)
    args = ap.parse_args()
    segs = json.loads((DATASET / "meta" / f"segments_{args.kind}.json").read_text(encoding="utf-8"))
    out_path = DATASET / "meta" / f"dedup_{args.kind}.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    ref = load_reference()
    print(f"reference: {len(ref)} Snatch 1.0 videos", flush=True)
    fp_dir = DATASET / "meta" / f"fp_web_{args.kind}"

    todo = []
    for vid, ss in segs.items():
        path = DATASET / "raw" / args.kind / f"{vid}.mp4"
        cap = cv2.VideoCapture(str(path))
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        for i, s in enumerate(ss):
            if s.get("error") or not s.get("static") or s.get("replay_of_other") or s["end"] - s["start"] < args.min_seconds:
                continue
            todo.append((-h, -(s["end"] - s["start"]), f"{vid}#{i}", path, s))
    todo.sort(key=lambda x: (x[0], x[1]))

    accepted: dict[str, F.Fingerprint] = {}
    for key, r in results.items():
        if r["status"] in ("ok", "check") and (fp_dir / f"{key.replace('#', '_')}.npz").exists():
            accepted[key] = F.load(fp_dir / f"{key.replace('#', '_')}.npz")
    for n, (_, _, key, path, s) in enumerate(todo, 1):
        if key in results:
            continue
        fp = F.compute(path, max_seconds=s["end"] - s["start"], start_s=s["start"])
        if fp is None or len(fp.app) < 4:
            results[key] = {"status": "unreadable"}
            continue
        F.save(fp, fp_dir / f"{key.replace('#', '_')}.npz")
        status, match, best = "ok", None, {"motion": 0, "appearance": 0}
        for name, rfp in ref.items():
            sim = F.similarity(fp, rfp)
            if sim["same_footage"]:
                status, match, best = "dup_snatch1", name, sim
                break
            if sim["possible"] and status == "ok":
                status, match, best = "check", f"snatch1:{name}", sim
        if status in ("ok", "check"):
            for other, ofp in accepted.items():
                if other.split("#")[0] == key.split("#")[0]:
                    continue
                sim = F.similarity(fp, ofp)
                if sim["same_footage"]:
                    status, match, best = "dup_web", other, sim
                    break
        results[key] = {"status": status, "match": match, "scores": best, "start": s["start"], "end": s["end"]}
        if status in ("ok", "check"):
            accepted[key] = fp
        print(f"[{n}/{len(todo)}] {key}: {status}{' ' + str(match) if match else ''}", flush=True)
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("summary:", counts)


if __name__ == "__main__":
    main()

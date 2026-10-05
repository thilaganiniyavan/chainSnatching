"""Stage 4: pre-screen, review sheets, and the final dataset build.

    python -m snatchguard.webdata.review prescreen --kind snatch      # detector counts per segment
    python -m snatchguard.webdata.review sheets --kind snatch          # coarse review images (4 segments each)
    python -m snatchguard.webdata.review fine KEY START END             # 12 frames around an event
    python -m snatchguard.webdata.review build                         # cut clips, write labels + manifest

Every clip in the dataset is reviewed by a person looking at its frames: real fixed-camera footage
(not staged, animated or AI-generated), the kind of snatching, lighting, and when it happens.
Decisions live in review/decisions_<kind>.csv:
    key,decision,lighting,event_start_s,event_end_s,notes
    decision: snatch_bike | snatch_foot | normal | no_event | not_cctv | staged_or_ai | unclear
    times are seconds in the *source video* (the build step converts them to clip time)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from snatchguard.webdata.discover import DATASET

REVIEW = DATASET / "review"


def _segments(kind: str, statuses=("ok", "check")) -> list[tuple[str, dict]]:
    dd = json.loads((DATASET / "meta" / f"dedup_{kind}.json").read_text(encoding="utf-8"))
    return [(k, v) for k, v in dd.items() if v["status"] in statuses]


def prescreen(kind: str) -> None:
    """People / two-wheelers seen in each segment (2 fps) with the laptop detector."""
    from ultralytics import YOLO
    from snatchguard.core.config import REPO_ROOT
    model = YOLO(str(REPO_ROOT / "models" / "yolo26n_512_openvino_model"), task="detect")
    out = REVIEW / f"prescreen_{kind}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for n, (key, seg) in enumerate(_segments(kind), 1):
        if key in res:
            continue
        cap = cv2.VideoCapture(str(DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        persons, bikes, frames = [], [], 0
        t = seg["start"]
        while t <= seg["end"]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                break
            r = model.predict(img, imgsz=512, conf=0.25, classes=[0, 1, 3], verbose=False)[0]
            c = r.boxes.cls.int().tolist() if r.boxes is not None else []
            persons.append(sum(1 for x in c if x == 0))
            bikes.append(sum(1 for x in c if x in (1, 3)))
            frames += 1
            t += 0.5
        res[key] = {"frames": frames, "max_persons": max(persons, default=0), "max_bikes": max(bikes, default=0),
                    "frames_with_person": sum(1 for x in persons if x), "frames_with_bike": sum(1 for x in bikes if x)}
        print(f"[{n}] {key}: {res[key]}", flush=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")


def _tile(img: np.ndarray, label: str, w: int = 320, h: int = 180) -> np.ndarray:
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(t, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return t


def frames_at(path: Path, times: list[float]) -> list[np.ndarray | None]:
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        out.append(img if ok else None)
    return out


def sheets(kind: str, per_sheet: int = 4, tiles: int = 6) -> None:
    """Coarse review images: `per_sheet` segments per image, `tiles` frames per segment."""
    pre = json.loads((REVIEW / f"prescreen_{kind}.json").read_text(encoding="utf-8"))
    segs = [(k, v) for k, v in _segments(kind) if pre.get(k, {}).get("frames_with_person", 0) >= 2]
    sheet_dir = REVIEW / f"sheets_{kind}"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for s in range(0, len(segs), per_sheet):
        rows = []
        for key, seg in segs[s:s + per_sheet]:
            ts = list(np.linspace(seg["start"], seg["end"] - 0.1, tiles))
            imgs = frames_at(DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4", ts)
            row = [_tile(im if im is not None else np.zeros((180, 320, 3), np.uint8),
                         f"{key} {t:.1f}s" if i == 0 else f"{t:.1f}s") for i, (im, t) in enumerate(zip(imgs, ts))]
            rows.append(np.hstack(row))
            index.append({"sheet": s // per_sheet, "key": key, **seg, **pre.get(key, {})})
        cv2.imwrite(str(sheet_dir / f"sheet_{s // per_sheet:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    (sheet_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(segs)} segments with people -> {(len(segs) + per_sheet - 1) // per_sheet} sheets in {sheet_dir}")


def fine(kind: str, key: str, start: float, end: float, out: Path) -> None:
    ts = list(np.linspace(start, end, 12))
    imgs = frames_at(DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4", ts)
    tiles = [_tile(im if im is not None else np.zeros((180, 320, 3), np.uint8), f"{t:.2f}s", 400, 225)
             for im, t in zip(imgs, ts)]
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]))


def build() -> None:
    """Cut every accepted segment into clips/, write labels.csv (SnatchGuard format) and manifest.csv."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    labels, manifest = [], []
    for kind in ("snatch", "normal"):
        dec_path = REVIEW / f"decisions_{kind}.csv"
        if not dec_path.exists():
            continue
        dd = json.loads((DATASET / "meta" / f"dedup_{kind}.json").read_text(encoding="utf-8"))
        with open(dec_path, newline="", encoding="utf-8") as f:
            decisions = list(csv.DictReader(f))
        for i, d in enumerate(decisions):
            if d["decision"] not in ("snatch_bike", "snatch_foot", "normal"):
                continue
            key = d["key"]
            vid = key.split("#")[0]
            seg = dd[key]
            # snatch clips: the segment trimmed to at most 20 s before and 10 s after the event
            start, end = seg["start"], seg["end"]
            if d["decision"] != "normal" and d["event_start_s"]:
                start = max(start, float(d["event_start_s"]) - 20.0)
                end = min(end, float(d["event_end_s"] or d["event_start_s"]) + 10.0)
            sub = "normal" if d["decision"] == "normal" else "snatch"
            name = f"{vid}_{key.split('#')[1]}.mp4"
            out = DATASET / "clips" / sub / name
            if not out.exists():
                subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                                "-i", str(DATASET / "raw" / kind / f"{vid}.mp4"), "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
            info = json.loads((DATASET / "raw" / kind / f"{vid}.info.json").read_text(encoding="utf-8"))
            es = f"{float(d['event_start_s']) - start:.2f}" if d["event_start_s"] else ""
            ee = f"{float(d['event_end_s']) - start:.2f}" if d["event_end_s"] else ""
            fold = "A" if len([x for x in labels if x["category"] == d["decision"]]) % 2 == 0 else "B"
            labels.append({"video": f"{sub}/{name}", "category": d["decision"], "lighting": d["lighting"],
                           "event_start_s": es, "event_end_s": ee, "fold": fold, "notes": d.get("notes", "")})
            manifest.append({"clip": f"clips/{sub}/{name}", "category": d["decision"], "lighting": d["lighting"],
                             "source_url": info.get("webpage_url"), "source_id": vid, "channel": info.get("channel"),
                             "title": info.get("title"), "upload_date": info.get("upload_date"),
                             "license": info.get("license") or "standard YouTube licence",
                             "source_start_s": f"{start:.2f}", "source_end_s": f"{end:.2f}",
                             "event_start_s": es, "event_end_s": ee,
                             "dedup": seg["status"], "dedup_scores": json.dumps(seg.get("scores", {}))})
    for name, rows in (("labels.csv", labels), ("manifest.csv", manifest)):
        if rows:
            with open(DATASET / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{sum(1 for x in labels if x['category'] != 'normal')} snatch clips, "
          f"{sum(1 for x in labels if x['category'] == 'normal')} normal clips -> {DATASET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prescreen", "sheets"):
        s = sub.add_parser(name)
        s.add_argument("--kind", choices=["snatch", "normal"], required=True)
    fn = sub.add_parser("fine")
    fn.add_argument("key")
    fn.add_argument("start", type=float)
    fn.add_argument("end", type=float)
    fn.add_argument("--kind", default="snatch")
    fn.add_argument("--out", type=Path, default=REVIEW / "fine.jpg")
    sub.add_parser("build")
    args = ap.parse_args()
    if args.cmd == "prescreen":
        prescreen(args.kind)
    elif args.cmd == "sheets":
        sheets(args.kind)
    elif args.cmd == "fine":
        fine(args.kind, args.key, args.start, args.end, args.out)
    else:
        build()


if __name__ == "__main__":
    main()

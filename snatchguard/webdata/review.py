"""Stage 4: pre-screen, review sheets, and the final dataset build.

    python -m snatchguard.webdata.review prescreen --kind snatch      # detector counts per segment
    python -m snatchguard.webdata.review sheets --kind snatch          # coarse review images (4 segments each)
    python -m snatchguard.webdata.review fine KEY START END             # 12 frames around an event
    python -m snatchguard.webdata.review fine-batch --kind snatch      # fine images for every 'unclear' decision
    python -m snatchguard.webdata.review build                         # cut clean clips, write labels + manifest

Every clip in the dataset is reviewed by a person looking at its frames: real fixed-camera footage
(not staged, animated or AI-generated), the kind of snatching, lighting, and when it happens.
Decisions live in review/decisions_<kind>.csv:
    key,decision,lighting,event_start_s,event_end_s,notes
    decision: snatch_bike | snatch_foot | normal | no_event | not_cctv | staged_or_ai | unclear
    times are seconds in the *source video* (the build step converts them to clip time)
Fine decisions (review/fine_decisions_<kind>.csv) override coarse ones and add clip bounds and `clean`.

Clean-only rule (what goes into labels.csv): the footage must look like it does on the camera itself.
  * no moving or blinking graphics (circles, arrows, spotlights, rotating ads), no zoom/crop change,
    no slow motion or loop, no picture-in-picture, not a phone filming a screen at an angle
  * static logos / tickers / DVR bars at the frame edge are allowed (they only hide a fixed border)
  * overlay score below overlay.OVERLAY_SCORE (automatic) and `clean` = yes (human)
  * snatch: at least MIN_LEAD_S of footage before the event starts, so tracks can form
  * normal: at least MIN_NORMAL_S long
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from snatchguard.webdata.discover import DATASET, source_of

REVIEW = DATASET / "review"
MIN_LEAD_S = 5.0          # footage before a snatch starts
MIN_NORMAL_S = 60.0


def _segments(kind: str, statuses=("ok", "check")) -> list[tuple[str, dict]]:
    dd = json.loads((DATASET / "meta" / f"dedup_{kind}.json").read_text(encoding="utf-8"))
    return [(k, v) for k, v in dd.items() if v["status"] in statuses]


def prescreen(kind: str) -> None:
    """People / two-wheelers seen in each segment (2 fps) with the laptop detector."""
    from ultralytics import YOLO
    from snatchguard.core.config import REPO_ROOT
    from snatchguard.webdata.overlay import sample_frames
    model = YOLO(str(REPO_ROOT / "models" / "yolo26n_512_openvino_model"), task="detect")
    out = REVIEW / f"prescreen_{kind}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    min_len = MIN_NORMAL_S if kind == "normal" else 0.0     # shorter normal shots can't make a clean clip
    for n, (key, seg) in enumerate(_segments(kind), 1):
        if key in res or seg["end"] - seg["start"] < min_len:
            continue
        persons, bikes, heights, frames = [], [], [], 0
        for _, img in sample_frames(DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4", seg["start"], seg["end"],
                                    1.0, max_samples=180):
            r = model.predict(img, imgsz=512, conf=0.25, classes=[0, 1, 3], verbose=False)[0]
            c = r.boxes.cls.int().tolist() if r.boxes is not None else []
            xyxy = r.boxes.xyxy.tolist() if r.boxes is not None else []
            persons.append(sum(1 for x in c if x == 0))
            bikes.append(sum(1 for x in c if x in (1, 3)))
            heights += [(b[3] - b[1]) / img.shape[0] for b, x in zip(xyxy, c) if x == 0]
            frames += 1
        heights.sort()
        res[key] = {"frames": frames, "max_persons": max(persons, default=0), "max_bikes": max(bikes, default=0),
                    "frames_with_person": sum(1 for x in persons if x), "frames_with_bike": sum(1 for x in bikes if x),
                    # CCTV shows people small in a wide view; a news anchor shot shows one large person
                    "median_person_height": round(heights[len(heights) // 2], 3) if heights else None}
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


def reviewed(kind: str) -> set[str]:
    keys = set()
    for name in (f"decisions_{kind}.csv", f"fine_decisions_{kind}.csv"):
        if (REVIEW / name).exists():
            with open(REVIEW / name, newline="", encoding="utf-8") as f:
                keys |= {r["key"] for r in csv.DictReader(f)}
    return keys


def _overlay(kind: str) -> dict:
    p = DATASET / "meta" / f"overlay_{kind}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def candidates(kind: str, per_video: int = 2) -> list[tuple[str, dict]]:
    """Unreviewed segments that pass the pre-screen and are long enough to make a clean clip."""
    return select(kind, per_video, new_only=True, use_overlay=False)


def select(kind: str, per_video: int = 2, new_only: bool = False, use_overlay: bool = True) -> list[tuple[str, dict]]:
    """CCTV-like segments with people in them, best `per_video` per source video.
    new_only: skip segments already reviewed, annotated ones (overlay score) and ones too short to be clean."""
    from snatchguard.webdata.overlay import OVERLAY_SCORE
    pre = json.loads((REVIEW / f"prescreen_{kind}.json").read_text(encoding="utf-8"))
    done = reviewed(kind) if new_only else set()
    ov = _overlay(kind) if new_only and use_overlay else {}
    min_len = MIN_LEAD_S + 3.0 if kind == "snatch" else MIN_NORMAL_S
    by_vid: dict[str, list] = {}
    for k, v in _segments(kind):
        if k in done:
            continue
        if new_only and v["end"] - v["start"] < min_len:
            continue
        if new_only and use_overlay:
            sc = ov.get(k, {}).get("score")
            if sc is None or sc >= OVERLAY_SCORE:
                continue
        p = pre.get(k)
        if not p or p["frames_with_person"] < 2 or (p["median_person_height"] or 1) > 0.45:
            continue
        if kind == "snatch" and p["frames_with_bike"] == 0 and p["max_persons"] < 2:
            continue                  # a snatching needs a bike, or at least two people
        score = (2 * p["frames_with_bike"] + p["frames_with_person"]) / max(p["frames"], 1)
        by_vid.setdefault(k.split("#")[0], []).append((score, k, v))
    out = []
    for vid, items in by_vid.items():
        for _, k, v in sorted(items, reverse=True)[:per_video]:
            out.append((k, v))
    return sorted(out)


def sheets(kind: str, per_sheet: int = 4, tiles: int = 6, new_only: bool = False, tag: str = "") -> None:
    """Coarse review images: `per_sheet` segments per image, `tiles` frames per segment."""
    pre = json.loads((REVIEW / f"prescreen_{kind}.json").read_text(encoding="utf-8"))
    segs = select(kind, new_only=new_only)
    sheet_dir = REVIEW / f"sheets_{kind}{'_' + tag if tag else ''}"
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


def fine(kind: str, key: str, start: float, end: float, out: Path, n: int = 16) -> None:
    """`n` frames (a multiple of 4) between start and end, 4 per row, labelled with source-video time."""
    ts = list(np.linspace(start, end, n))
    imgs = frames_at(DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4", ts)
    tiles = [_tile(im if im is not None else np.zeros((180, 320, 3), np.uint8),
                 f"{key} {t:.2f}s" if i == 0 else f"{t:.2f}s", 400, 225)
             for i, (im, t) in enumerate(zip(imgs, ts))]
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def fine_batch(kind: str, decision: str = "unclear") -> None:
    """A fine image for every segment with the given decision, over its noted event window."""
    out_dir = REVIEW / f"fine_{kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dd = json.loads((DATASET / "meta" / f"dedup_{kind}.json").read_text(encoding="utf-8"))
    rows = [dict(r, key=k) for k, r in decisions(kind).items() if r["decision"] == decision]
    for r in rows:
        seg = dd[r["key"]]
        s = float(r.get("event_start_s") or seg["start"])
        e = float(r.get("event_end_s") or seg["end"])
        s, e = max(seg["start"], s - 1.0), min(seg["end"], e + 1.0)
        fine(kind, r["key"], s, e, out_dir / f"{r['key'].replace('#', '_')}.jpg")
    print(f"{len(rows)} fine images -> {out_dir}")


def decisions(kind: str) -> dict[str, dict]:
    """Coarse decisions overridden by fine ones (which may add clip bounds and `clean`)."""
    out: dict[str, dict] = {}
    for name in (f"decisions_{kind}.csv", f"fine_decisions_{kind}.csv"):
        if (REVIEW / name).exists():
            with open(REVIEW / name, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out[r["key"]] = r
    return out


def build(max_normal_s: float = 600.0) -> None:
    """Cut every accepted *clean* segment into clips/, write labels.csv (SnatchGuard format) and manifest.csv."""
    import imageio_ffmpeg
    from snatchguard.webdata.overlay import OVERLAY_SCORE
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    labels, manifest, rejected = [], [], {}
    for kind in ("snatch", "normal"):
        dd_path = DATASET / "meta" / f"dedup_{kind}.json"
        if not dd_path.exists():
            continue
        dd = json.loads(dd_path.read_text(encoding="utf-8"))
        ov = _overlay(kind)
        for key, d in sorted(decisions(kind).items()):
            if d["decision"] not in ("snatch_bike", "snatch_foot", "normal"):
                continue
            if (d.get("clean") or "").lower() != "yes":
                rejected["not marked clean"] = rejected.get("not marked clean", 0) + 1
                continue
            seg = dd[key]
            start = float(d.get("clip_start_s") or seg["start"])
            end = float(d.get("clip_end_s") or seg["end"])
            snatch = d["decision"] != "normal"
            if snatch and d["event_start_s"]:
                # at most 20 s before and 10 s after the event
                start = max(start, float(d["event_start_s"]) - 20.0)
                end = min(end, float(d["event_end_s"] or d["event_start_s"]) + 10.0)
            if not snatch:
                end = min(end, start + max_normal_s)
            score = (ov.get(key) or {}).get("score")
            reason = None
            if score is None:
                reason = "no overlay score"
            elif score >= OVERLAY_SCORE:
                reason = "moving graphics"
            elif snatch and (not d["event_start_s"] or float(d["event_start_s"]) - start < MIN_LEAD_S):
                reason = f"under {MIN_LEAD_S:.0f} s before the event"
            elif not snatch and end - start < MIN_NORMAL_S:
                reason = f"normal under {MIN_NORMAL_S:.0f} s"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            vid = key.split("#")[0]
            sub = "snatch" if snatch else "normal"
            name = f"{vid}_{key.split('#')[1]}.mp4"
            out = DATASET / "clips" / sub / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                                "-i", str(DATASET / "raw" / kind / f"{vid}.mp4"), "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
            info = json.loads((DATASET / "raw" / kind / f"{vid}.info.json").read_text(encoding="utf-8"))
            es = f"{float(d['event_start_s']) - start:.2f}" if snatch and d["event_start_s"] else ""
            ee = f"{float(d['event_end_s']) - start:.2f}" if snatch and d["event_end_s"] else ""
            fold = "A" if int(hashlib.md5(vid.encode()).hexdigest(), 16) % 2 == 0 else "B"
            labels.append({"video": f"{sub}/{name}", "category": d["decision"], "lighting": d["lighting"],
                           "event_start_s": es, "event_end_s": ee, "fold": fold, "notes": d.get("notes", "")})
            manifest.append({"clip": f"clips/{sub}/{name}", "category": d["decision"], "lighting": d["lighting"],
                             "source_url": info.get("webpage_url"), "source_id": vid, "channel": info.get("channel"),
                             "source_type": source_of({"channel": info.get("channel")}),
                             "title": info.get("title"), "upload_date": info.get("upload_date"),
                             "license": info.get("license") or "standard YouTube licence",
                             "source_start_s": f"{start:.2f}", "source_end_s": f"{end:.2f}",
                             "event_start_s": es, "event_end_s": ee, "overlay_score": score,
                             "dedup": seg["status"], "dedup_scores": json.dumps(seg.get("scores", {})),
                             "notes": d.get("notes", "")})
    for name, rows in (("labels.csv", labels), ("manifest.csv", manifest)):
        if rows:
            with open(DATASET / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    stats = {"clips": {}, "hours": {}, "lighting": {}, "source_type": {}, "rejected_after_review": rejected,
             "decisions": {}}
    for m in manifest:
        c = m["category"]
        stats["clips"][c] = stats["clips"].get(c, 0) + 1
        dur = float(m["source_end_s"]) - float(m["source_start_s"])
        stats["hours"][c] = round(stats["hours"].get(c, 0) + dur / 3600, 3)
        stats["lighting"][f"{c}/{m['lighting']}"] = stats["lighting"].get(f"{c}/{m['lighting']}", 0) + 1
        stats["source_type"][m["source_type"]] = stats["source_type"].get(m["source_type"], 0) + 1
    for kind in ("snatch", "normal"):
        for d in decisions(kind).values():
            stats["decisions"][d["decision"]] = stats["decisions"].get(d["decision"], 0) + 1
    (DATASET / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"{sum(1 for x in labels if x['category'] != 'normal')} snatch clips, "
          f"{sum(1 for x in labels if x['category'] == 'normal')} normal clips -> {DATASET}")
    print("accepted by review but not clean enough:", rejected)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prescreen", "sheets"):
        s = sub.add_parser(name)
        s.add_argument("--kind", choices=["snatch", "normal"], required=True)
        if name == "sheets":
            s.add_argument("--new-only", action="store_true", help="unreviewed, unannotated, long-enough segments")
            s.add_argument("--tag", default="", help="suffix for the sheet folder (keeps earlier sheets)")
    fn = sub.add_parser("fine")
    fn.add_argument("key")
    fn.add_argument("start", type=float)
    fn.add_argument("end", type=float)
    fn.add_argument("--kind", default="snatch")
    fn.add_argument("--out", type=Path, default=REVIEW / "fine.jpg")
    fb = sub.add_parser("fine-batch")
    fb.add_argument("--kind", default="snatch")
    fb.add_argument("--decision", default="unclear")
    sub.add_parser("build")
    args = ap.parse_args()
    if args.cmd == "prescreen":
        prescreen(args.kind)
    elif args.cmd == "sheets":
        sheets(args.kind, new_only=args.new_only, tag=args.tag)
    elif args.cmd == "fine-batch":
        fine_batch(args.kind, args.decision)
    elif args.cmd == "fine":
        fine(args.kind, args.key, args.start, args.end, args.out)
    else:
        build()


if __name__ == "__main__":
    main()

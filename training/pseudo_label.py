"""Teacher -> student distillation on your own camera footage (the domain the edge model must work in).

A large model (YOLO26x at 1280 px with test-time augmentation) is too slow for the edge but
far more accurate. On a Colab GPU it labels frames sampled from real CCTV recordings; those
labels are added to the detector training set so the small edge model learns the real
cameras' angle, compression, lighting and the local two-wheelers.

    python training/pseudo_label.py --videos /content/cctv --out /content/datasets/sg_det \\
        --every 2.0 --exclude s_2.avi s_3.avi

Hold out the videos you measure false alarms on (--exclude), otherwise the false-alarm
numbers are optimistic.
"""

from __future__ import annotations

import argparse
import random
import sys
import zlib
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.prepare_dataset import night_variant  # noqa: E402

TEACHER_TO_SG = {0: 0, 1: 1, 3: 2}      # COCO person, bicycle, motorcycle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path, help="folder with CCTV recordings (searched recursively)")
    ap.add_argument("--out", required=True, type=Path, help="detector dataset root (…/sg_det)")
    ap.add_argument("--teacher", default="yolo26x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.40, help="keep only confident teacher boxes")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--max-per-video", type=int, default=400)
    ap.add_argument("--exclude", nargs="*", default=[], help="file names to hold out for evaluation")
    ap.add_argument("--night-aug", action="store_true", help="also apply synthetic night variants")
    args = ap.parse_args()

    from ultralytics import YOLO
    teacher = YOLO(args.teacher)
    img_dir = args.out / "images" / "train"
    lbl_dir = args.out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    videos = [p for p in sorted(args.videos.rglob("*")) if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")
              and p.name not in args.exclude]
    total = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(fps * args.every))
        idx = kept = 0
        while kept < args.max_per_video:
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            r = teacher.predict(img, imgsz=args.imgsz, conf=args.conf, augment=True, classes=list(TEACHER_TO_SG),
                                verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            lines = [f"{TEACHER_TO_SG[int(c)]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                     for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())]
            name = f"pseudo_{v.stem}_{idx:07d}"
            if args.night_aug:
                img, kind = night_variant(img, random.Random(zlib.crc32(name.encode())))
                name = f"pseudo_{kind}_{v.stem}_{idx:07d}"
            h, w = img.shape[:2]
            if w > 1280:
                img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
        total += kept
        print(f"{v.name}: {kept} pseudo-labelled frames", flush=True)
    print(f"added {total} frames to {img_dir}")


if __name__ == "__main__":
    main()

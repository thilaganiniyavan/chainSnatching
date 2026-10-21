"""Teacher -> student distillation on your own camera footage (the domain the edge model must work in).

A large model (YOLO26x at 1280 px with test-time augmentation) is too slow for the edge but
far more accurate. On the training GPU it labels frames sampled from real CCTV recordings; those
labels are added to the detector training set so the small edge model learns the real
cameras' angle, compression, lighting and the local two-wheelers.

    python training/pseudo_label.py --out data/datasets/sg_det                     # Snatch 1.0 'pseudo' fold
    python training/pseudo_label.py --videos D:/my_cctv --out data/datasets/sg_det  # any folder of recordings

With the Snatch 1.0 labels, only the normal videos in the `pseudo` fold are used: the videos the
false-alarm and acceptance numbers are measured on are never seen by the detector.

A few whole videos (`--val-videos`) go to a `val_cctv` split instead of `train`, unaugmented: real CCTV
the detector is validated on (best.pt selection and `train.py compare`). Their labels come from the
teacher, so that number measures agreement with the teacher on real footage, not ground truth.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import zlib
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.prepare_dataset import night_variant, write_det_yaml  # noqa: E402

TEACHER_TO_SG = {0: 0, 1: 1, 3: 2}      # COCO person, bicycle, motorcycle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, help="folder with CCTV recordings (searched recursively); "
                    "default: Snatch 1.0 videos in the labels' pseudo fold")
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "labels" / "snatch1.0.csv")
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "samples" / "snatch1.0"
                    / "Chain Snatching Videos")
    ap.add_argument("--out", required=True, type=Path, help="detector dataset root (…/sg_det)")
    ap.add_argument("--teacher", default="yolo26x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.40, help="keep only confident teacher boxes")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--max-per-video", type=int, default=400)
    ap.add_argument("--exclude", nargs="*", default=[], help="file names to hold out for evaluation")
    ap.add_argument("--night-aug", action="store_true", help="also apply synthetic night variants")
    ap.add_argument("--val-videos", type=int, default=3, help="whole videos held out as the val_cctv split")
    args = ap.parse_args()

    from ultralytics import YOLO
    teacher = YOLO(args.teacher)
    for split in ("train", "val_cctv"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    if args.videos:
        videos = [p for p in sorted(args.videos.rglob("*")) if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")
                  and p.name not in args.exclude]
    else:
        import csv
        with open(args.labels, newline="", encoding="utf-8") as f:
            videos = [args.data / r["video"] for r in csv.DictReader(f) if r.get("fold") == "pseudo"]
        missing = [v for v in videos if not v.exists()]
        if missing:
            print(f"WARNING: {len(missing)} pseudo-fold videos not found (download/extract Snatch 1.0 first)")
        videos = [v for v in videos if v.exists()]
    # whole videos, chosen by a stable hash of the name, so reruns pick the same ones
    val_names = {v.name for v in sorted(videos, key=lambda p: zlib.crc32(p.name.encode()))[:args.val_videos]}
    if len(videos) <= args.val_videos:
        val_names = set()
    for v in videos:     # a validation video must have no frames left in train from an earlier run
        if v.name in val_names:
            own = re.compile(rf"^pseudo_(?:(?:day|ir|lowlight)_)?{re.escape(v.stem)}_\d{{7}}$")
            for sub in ("images", "labels"):
                for f in (args.out / sub / "train").glob("pseudo_*"):
                    if own.match(f.stem):
                        f.unlink()
    total = 0
    for v in videos:
        split = "val_cctv" if v.name in val_names else "train"
        img_dir, lbl_dir = args.out / "images" / split, args.out / "labels" / split
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
            if args.night_aug and split == "train":
                img, kind = night_variant(img, random.Random(zlib.crc32(name.encode())))
                name = f"pseudo_{kind}_{v.stem}_{idx:07d}"
            h, w = img.shape[:2]
            if w > 1280:
                img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
        total += kept
        print(f"{v.name}: {kept} pseudo-labelled frames -> {split}", flush=True)
    print(f"added {total} frames ({len(val_names)} videos held out as val_cctv: {sorted(val_names)})")
    if (args.out / "data.yaml").exists():
        write_det_yaml(args.out)


if __name__ == "__main__":
    main()

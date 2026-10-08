"""Fine-tune YOLO26 detector and pose models for night CCTV, then export for the edge.

    python training/train.py det  --data /content/datasets/sg_det  --epochs 60
    python training/train.py pose --data /content/datasets/sg_pose --epochs 30
    python training/train.py compare --data /content/datasets/sg_det --weights runs/det/best.pt

`compare` evaluates the pretrained COCO model and the fine-tuned model on the
day and night validation sets separately: the numbers to put in the report.
Only perception is trained. The snatch decision stays rule-based.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

# Augmentation tuned for CCTV: small motorcycles require careful handling.
# Mosaic stitching (combining 4 images) destroys small object detection.
# Strong augmentation on already-degraded CCTV creates distribution mismatch.
AUG = dict(hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, degrees=0.0, translate=0.05, scale=0.3, fliplr=0.5, flipud=0.0,
           mosaic=0.0, mixup=0.0, close_mosaic=0)


def train(kind: str, data: Path, epochs: int, imgsz: int, batch: int, size: str, project: Path, patience: int,
          workers: int = 4, device: str | None = None) -> Path:
    """Trains, or resumes automatically if a previous (disconnected) run left last.pt behind."""
    last = project.resolve() / kind / "weights" / "last.pt"
    if last.exists():
        print(f"resuming from {last}")
        model = YOLO(str(last))
        try:
            model.train(resume=True)
        except AssertionError as exc:          # previous run already finished all epochs
            print(f"nothing to resume ({exc})")
        return last.parent / "best.pt"
    base = f"yolo26{size}.pt" if kind == "det" else f"yolo26{size}-pose.pt"
    model = YOLO(base)
    extra = {"device": device} if device is not None else {}
    model.train(data=str(data / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, project=str(project.resolve()),
                name=kind, exist_ok=True, patience=patience, cos_lr=True, plots=True, workers=workers, **extra, **AUG)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")
    return best


def _coco_id_view(data: Path, split: str) -> Path:
    """Same val images, labels renumbered to COCO ids (motorcycle 2 -> 3), for scoring a COCO-pretrained model."""
    view = data / "_coco_ids"
    img_dst = view / "images" / f"val_{split}"
    lbl_dst = view / "labels" / f"val_{split}"
    if not img_dst.exists():
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        img_dst.symlink_to((data / "images" / f"val_{split}").resolve(), target_is_directory=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for f in (data / "labels" / f"val_{split}").glob("*.txt"):
            rows = [r.split() for r in f.read_text().splitlines() if r.strip()]
            (lbl_dst / f.name).write_text("".join(" ".join([{"2": "3"}.get(r[0], r[0])] + r[1:]) + "\n" for r in rows))
    yaml_path = view / f"data_{split}.yaml"
    names = {i: f"class{i}" for i in range(80)}
    names.update({0: "person", 1: "bicycle", 3: "motorcycle"})
    yaml_path.write_text(json.dumps({"path": str(view.resolve()), "train": f"images/val_{split}",
                                     "val": f"images/val_{split}", "names": names}))
    return yaml_path


def evaluate(weights: str, data: Path, imgsz: int, kind: str) -> dict:
    out = {}
    for split in ("day", "night"):
        yaml_path = data / f"data_{split}.yaml"
        if not yaml_path.exists():
            continue
        cfg = json.loads(yaml_path.read_text())
        if not cfg["val"]:
            continue
        m = YOLO(weights)
        if kind == "det":
            classes = None
            if len(m.names) == 80:        # COCO-pretrained: score only person / bicycle / motorcycle
                yaml_path, classes = _coco_id_view(data, split), [0, 1, 3]
            r = m.val(data=str(yaml_path), imgsz=imgsz, classes=classes, plots=False, verbose=False)
            out[split] = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4)}
        else:
            r = m.val(data=str(yaml_path), imgsz=imgsz, plots=False, verbose=False)
            out[split] = {"pose_mAP50": round(float(r.pose.map50), 4), "pose_mAP50-95": round(float(r.pose.map), 4)}
    return out


def export(weights: Path, imgsz: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(weights))
    for fmt in ("openvino", "onnx"):
        p = Path(m.export(format=fmt, imgsz=imgsz))
        if not p.is_absolute():
            p = weights.parent / p.name
        dest = out_dir / p.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(p), dest)
    shutil.copy(weights, out_dir / weights.name)
    print(f"exported to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["det", "pose", "compare"])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch (fits the GPU's memory, e.g. an 8 GB RTX 4060)")
    ap.add_argument("--size", default="n", choices=["n", "s", "m"])
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers (Windows: keep 2-4)")
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU (default: Ultralytics picks)")
    ap.add_argument("--export-to", type=Path, default=Path("export"))
    ap.add_argument("--weights", help="for compare: fine-tuned weights")
    ap.add_argument("--kind", choices=["det", "pose"], default="det", help="for compare")
    args = ap.parse_args()

    if args.task == "compare":
        base = f"yolo26{args.size}.pt" if args.kind == "det" else f"yolo26{args.size}-pose.pt"
        report = {"pretrained": evaluate(base, args.data, args.imgsz, args.kind)}
        if args.weights:
            report["finetuned"] = evaluate(args.weights, args.data, args.imgsz, args.kind)
        print(json.dumps(report, indent=2))
        (args.project / f"compare_{args.kind}.json").parent.mkdir(parents=True, exist_ok=True)
        (args.project / f"compare_{args.kind}.json").write_text(json.dumps(report, indent=2))
        return

    best = train(args.task, args.data, args.epochs, args.imgsz, args.batch, args.size, args.project, args.patience,
                 args.workers, args.device)
    export(best, 512 if args.task == "det" else 384, args.export_to / args.task)


if __name__ == "__main__":
    main()

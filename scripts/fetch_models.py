"""Download the pretrained YOLO26 detector + pose models and export them to OpenVINO.

    python scripts/fetch_models.py                 # yolo26n + yolo26n-pose -> models/
    python scripts/fetch_models.py --size s        # bigger / more accurate
    python scripts/fetch_models.py --from-colab path/to/best_det.pt path/to/best_pose.pt
    python scripts/fetch_models.py --verifier       # also the video network of the two-stage verifier

OpenVINO is the fastest CPU backend for Intel and AMD laptops. On a Jetson,
export with format=engine (TensorRT) on the device itself instead.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def export_openvino(weights: Path, imgsz: int) -> Path:
    """Ultralytics writes <stem>_openvino_model/ next to the weights."""
    target = weights.parent / f"{weights.stem}_openvino_model"
    if target.exists():
        shutil.rmtree(target)
    YOLO(str(weights)).export(format="openvino", imgsz=imgsz, half=False, dynamic=False)
    if not target.exists():
        raise RuntimeError(f"OpenVINO export did not produce {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="n", choices=["n", "s", "m"])
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--pose-imgsz", type=int, default=384)
    ap.add_argument("--from-colab", nargs=2, metavar=("DET_PT", "POSE_PT"),
                    help="use fine-tuned weights from training/ instead of the pretrained ones")
    ap.add_argument("--no-export", action="store_true", help="keep .pt only (slower on CPU)")
    ap.add_argument("--verifier", action="store_true",
                    help="also download the Kinetics R(2+1)D-18 weights the two-stage verifier needs (needs torchvision)")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    if args.verifier:
        from torchvision.models.video import R2Plus1D_18_Weights
        R2Plus1D_18_Weights.KINETICS400_V1.get_state_dict(progress=True)   # cached under ~/.cache/torch
        print("verifier backbone: Kinetics-400 R(2+1)D-18 weights cached")

    if args.from_colab:
        det_pt, pose_pt = MODELS / "det_finetuned.pt", MODELS / "pose_finetuned.pt"
        shutil.copy(args.from_colab[0], det_pt)
        shutil.copy(args.from_colab[1], pose_pt)
    else:
        det_pt, pose_pt = MODELS / f"yolo26{args.size}.pt", MODELS / f"yolo26{args.size}-pose.pt"
        for target in (det_pt, pose_pt):
            if not target.exists():
                YOLO(target.name)                      # Ultralytics downloads into the CWD
                shutil.move(target.name, target)

    print(f"weights: {det_pt}, {pose_pt}")
    if args.no_export:
        return
    det_ov = export_openvino(det_pt, args.det_imgsz)
    pose_ov = export_openvino(pose_pt, args.pose_imgsz)
    print("\nSet these in configs/edge.yaml under models:")
    print(f"  detector: {det_ov.relative_to(ROOT)}")
    print(f"  pose: {pose_ov.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

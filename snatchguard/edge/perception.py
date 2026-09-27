"""Detection + tracking + on-demand pose, on Ultralytics YOLO26 models.

* The detector runs on every processed frame and tracks people, bicycles and
  motorcycles with OC-SORT.
* Pose runs only on the few regions the rule engine asks for (people close to
  a bike), cropped from the full-resolution frame, so a pillion rider's wrist
  is still visible on a wide CCTV view. Everyone else never costs a pose pass.

Any Ultralytics-loadable model works: `.pt`, an OpenVINO export directory
(fastest on Intel/AMD CPUs), ONNX or TensorRT. Class ids are resolved from the
model's own names, so fine-tuned night models with different ids just work.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.types import Detection, ObjClass, Pose

log = logging.getLogger(__name__)

_NAME_MAP = {
    "person": ObjClass.PERSON, "people": ObjClass.PERSON, "pedestrian": ObjClass.PERSON,
    "bicycle": ObjClass.BICYCLE, "bike": ObjClass.BICYCLE,
    "motorcycle": ObjClass.MOTORCYCLE, "motorbike": ObjClass.MOTORCYCLE, "scooter": ObjClass.MOTORCYCLE,
}


def _load(path: str, task: str):
    from ultralytics import YOLO  # heavy import, only on the edge

    p = Path(path)
    if not p.exists() and not path.endswith(".pt"):
        raise FileNotFoundError(f"model not found: {path} (run scripts/fetch_models.py)")
    return YOLO(str(path), task=task)


class Perception:
    def __init__(self, cfg: dict, tracker_cfg: str) -> None:
        self.cfg = cfg
        self.tracker_cfg = tracker_cfg
        self.det = _load(cfg["detector"], "detect")
        self.pose = _load(cfg["pose"], "pose") if cfg.get("pose") else None
        self.imgsz = int(cfg.get("imgsz", 640))
        self.pose_imgsz = int(cfg.get("pose_imgsz", 384))
        self.device = cfg.get("device", "cpu")
        self.class_of: dict[int, ObjClass] = {}
        for cid, name in self.det.names.items():
            cls = _NAME_MAP.get(str(name).lower())
            if cls is not None:
                self.class_of[int(cid)] = cls
        if ObjClass.PERSON not in self.class_of.values():
            raise ValueError(f"detector has no person class: {self.det.names}")
        log.info("detector classes used: %s", {k: v.value for k, v in self.class_of.items()})
        self.det_ms = 0.0
        self.pose_ms = 0.0

    def detect_track(self, img: np.ndarray, conf: float) -> list[Detection]:
        t0 = time.perf_counter()
        res = self.det.track(img, persist=True, tracker=self.tracker_cfg, conf=conf, iou=0.6, imgsz=self.imgsz,
                             classes=list(self.class_of), device=self.device, verbose=False)[0]
        self.det_ms = (time.perf_counter() - t0) * 1000
        boxes = res.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        cf = boxes.conf.cpu().numpy()
        return [Detection(int(i), self.class_of[int(c)], b.astype(float), float(s))
                for b, i, c, s in zip(xyxy, ids, cls, cf) if int(c) in self.class_of]

    def estimate_poses(self, img: np.ndarray, crops: list[np.ndarray],
                       persons: dict[int, np.ndarray]) -> dict[int, Pose]:
        """Pose inside each crop, matched back to tracked people by IoU."""
        if self.pose is None or not crops:
            self.pose_ms = 0.0
            return {}
        t0 = time.perf_counter()
        found: list[tuple[np.ndarray, Pose]] = []
        for crop in crops:
            x1, y1, x2, y2 = (int(v) for v in crop)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            res = self.pose.predict(img[y1:y2, x1:x2], imgsz=self.pose_imgsz, conf=0.2, device=self.device,
                                    verbose=False)[0]
            if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
                continue
            kxy = res.keypoints.xy.cpu().numpy()
            kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
            bxy = res.boxes.xyxy.cpu().numpy()
            off = np.array([x1, y1], dtype=float)
            for k, c, b in zip(kxy, kcf, bxy):
                found.append((b + np.array([x1, y1, x1, y1], dtype=float), Pose(k + off, c.astype(float))))
        self.pose_ms = (time.perf_counter() - t0) * 1000

        out: dict[int, Pose] = {}
        pairs = sorted(((g.iou(pb, tb), tid, i) for i, (pb, _) in enumerate(found) for tid, tb in persons.items()),
                       reverse=True)
        used_t, used_p = set(), set()
        for score, tid, i in pairs:
            if score < 0.3:
                break
            if tid in used_t or i in used_p:
                continue
            out[tid] = found[i][1]
            used_t.add(tid)
            used_p.add(i)
        return out

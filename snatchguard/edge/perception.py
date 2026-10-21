"""Detection + tracking + on-demand pose, on Ultralytics YOLO26 models.

* The detector runs on every processed frame and tracks people, bicycles and
  motorcycles with OC-SORT.
* Pose runs only on the few regions the rule engine asks for (people close to
  a bike), cropped from the full-resolution frame, so a pillion rider's wrist
  is still visible on a wide CCTV view. Everyone else never costs a pose pass.
* Optional tiled detection (`tiles: [rows, cols]`, `tile_overlap`): the
  detector also runs on overlapping tiles at full input size, so small, distant
  motorcycles are seen at a usable scale. Tile and full-frame boxes are merged
  and then tracked with the same tracker config.

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


def resolve_device(device: str | int | None) -> str | int:
    """'auto' -> first CUDA GPU if PyTorch sees one, else CPU."""
    if device in (None, "auto"):
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


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
        self.device = resolve_device(cfg.get("device", "auto"))
        if "openvino" in str(cfg["detector"]).lower():
            self.device = "cpu"                     # OpenVINO exports run on the CPU (Intel GPU not targeted)
        # FP16 only for PyTorch weights on a GPU (OpenVINO / ONNX / TensorRT exports carry their own precision)
        self.half = self.device != "cpu" and str(cfg["detector"]).endswith(".pt")
        self._prec = {"quantize": 16} if self.half else {}
        # "crops": pose only around the closest pairs (CPU). "full": pose on the whole frame (GPU).
        self.pose_mode = cfg.get("pose_mode", "full" if self.device != "cpu" else "crops")
        self.class_of: dict[int, ObjClass] = {}
        for cid, name in self.det.names.items():
            cls = _NAME_MAP.get(str(name).lower())
            if cls is not None:
                self.class_of[int(cid)] = cls
        if ObjClass.PERSON not in self.class_of.values():
            raise ValueError(f"detector has no person class: {self.det.names}")
        log.info("detector classes used: %s", {k: v.value for k, v in self.class_of.items()})
        self.tiles = tuple(int(v) for v in cfg["tiles"]) if cfg.get("tiles") else None
        self.tile_overlap = float(cfg.get("tile_overlap", 0.2))
        self._tracker = None
        self.det_ms = 0.0
        self.pose_ms = 0.0

    def reset_tracking(self) -> None:
        """Forget all tracks (new scene), whichever tracking path is in use."""
        if self._tracker is not None:
            self._tracker.reset()
        predictor = getattr(self.det, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            tr.reset()

    def detect_track(self, img: np.ndarray, conf: float) -> list[Detection]:
        if self.tiles:
            return self._detect_track_tiled(img, conf)
        t0 = time.perf_counter()
        res = self.det.track(img, persist=True, tracker=self.tracker_cfg, conf=conf, iou=0.6, imgsz=self.imgsz,
                             classes=list(self.class_of), device=self.device, verbose=False, **self._prec)[0]
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

    def detect_tiled(self, img: np.ndarray, conf: float) -> np.ndarray:
        """Untracked detections from the full frame plus every tile, merged: rows (x1, y1, x2, y2, conf, cls)."""
        h, w = img.shape[:2]
        parts = []
        for ri, (x1, y1, x2, y2) in enumerate([(0, 0, w, h), *tile_regions(w, h, *self.tiles, self.tile_overlap)]):
            r = self.det.predict(img[y1:y2, x1:x2], conf=conf, iou=0.6, imgsz=self.imgsz, classes=list(self.class_of),
                                 device=self.device, verbose=False, **self._prec)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.data.cpu().numpy()[:, :6].astype(float)
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            if ri:
                b = b[~_touches_inner_edge(b, (x1, y1, x2, y2), w, h)]
            parts.append(np.column_stack([b, np.full(len(b), ri)]))
        if not parts:
            return np.zeros((0, 6))
        return merge_tiled(np.concatenate(parts))

    def _detect_track_tiled(self, img: np.ndarray, conf: float) -> list[Detection]:
        from ultralytics.engine.results import Boxes

        t0 = time.perf_counter()
        if self._tracker is None:
            self._tracker = _make_tracker(self.tracker_cfg)
        dets = self.detect_tiled(img, conf)
        tracks = self._tracker.update(Boxes(dets.astype(np.float32), img.shape[:2]), img)
        self.det_ms = (time.perf_counter() - t0) * 1000
        # rows: x1, y1, x2, y2, track_id, conf, cls, det_index
        return [Detection(int(t[4]), self.class_of[int(t[6])], t[:4].astype(float), float(t[5]))
                for t in tracks if int(t[6]) in self.class_of]

    def estimate_poses(self, img: np.ndarray, crops: list[np.ndarray],
                       persons: dict[int, np.ndarray]) -> dict[int, Pose]:
        """Pose inside each crop (cropped from the full-resolution frame), matched back to tracked people by IoU."""
        if self.pose is None or not crops:
            self.pose_ms = 0.0
            return {}
        t0 = time.perf_counter()
        found: list[tuple[np.ndarray, Pose]] = []
        for crop in crops:
            x1, y1, x2, y2 = (int(v) for v in crop)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            found += self._pose_pass(img[y1:y2, x1:x2], (x1, y1))
        self.pose_ms = (time.perf_counter() - t0) * 1000
        return self._match(found, persons)

    def estimate_poses_full(self, img: np.ndarray, persons: dict[int, np.ndarray]) -> dict[int, Pose]:
        """One pose pass over the whole frame (GPU): keypoints for everyone at once."""
        if self.pose is None or not persons:
            self.pose_ms = 0.0
            return {}
        t0 = time.perf_counter()
        found = self._pose_pass(img, (0, 0))
        self.pose_ms = (time.perf_counter() - t0) * 1000
        return self._match(found, persons)

    def _pose_pass(self, img: np.ndarray, offset: tuple[int, int]) -> list[tuple[np.ndarray, Pose]]:
        res = self.pose.predict(img, imgsz=self.pose_imgsz, conf=0.2, device=self.device, verbose=False,
                                **self._prec)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        bxy = res.boxes.xyxy.cpu().numpy()
        ox, oy = offset
        return [(b + np.array([ox, oy, ox, oy], dtype=float), Pose(k + np.array([ox, oy], dtype=float), c.astype(float)))
                for k, c, b in zip(kxy, kcf, bxy)]

    @staticmethod
    def _match(found: list[tuple[np.ndarray, Pose]], persons: dict[int, np.ndarray]) -> dict[int, Pose]:
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


# ---- tiled detection helpers ------------------------------------------------------------------------

def tile_regions(w: int, h: int, rows: int, cols: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the frame: `overlap` is the fraction of a tile shared with its neighbour."""
    def spans(size: int, n: int) -> list[tuple[int, int]]:
        if n <= 1:
            return [(0, size)]
        tile = size / (n - (n - 1) * overlap)
        step = tile * (1 - overlap)
        return [(int(round(i * step)), size if i == n - 1 else int(round(i * step + tile))) for i in range(n)]
    return [(x1, y1, x2, y2) for y1, y2 in spans(h, rows) for x1, x2 in spans(w, cols)]


def _touches_inner_edge(b: np.ndarray, region: tuple[int, int, int, int], w: int, h: int, margin: float = 2.0) -> np.ndarray:
    """Boxes cut by a tile border that is not a frame border. Such an object is either whole in a
    neighbouring tile (it is smaller than the overlap) or large enough for the full-frame pass."""
    x1, y1, x2, y2 = region
    return (((x1 > 0) & (b[:, 0] <= x1 + margin)) | ((y1 > 0) & (b[:, 1] <= y1 + margin))
            | ((x2 < w) & (b[:, 2] >= x2 - margin)) | ((y2 < h) & (b[:, 3] >= y2 - margin)))


def merge_tiled(rows: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy class-wise merge of boxes from different passes (rows: x1, y1, x2, y2, conf, cls, region).

    Within one pass the model has already removed duplicates, so only boxes from different passes are
    compared. Plain IoU on purpose: boxes cut by a tile border are dropped before this, and a person
    standing inside another person's box (the moment of a grab) must stay a separate detection."""
    order = np.argsort(-rows[:, 4], kind="stable")
    kept: list[int] = []
    area = (rows[:, 2] - rows[:, 0]).clip(0) * (rows[:, 3] - rows[:, 1]).clip(0)
    for i in order:
        dup = False
        for k in kept:
            if rows[k, 5] != rows[i, 5] or rows[k, 6] == rows[i, 6]:
                continue
            iw = min(rows[i, 2], rows[k, 2]) - max(rows[i, 0], rows[k, 0])
            ih = min(rows[i, 3], rows[k, 3]) - max(rows[i, 1], rows[k, 1])
            if iw > 0 and ih > 0 and iw * ih / (area[i] + area[k] - iw * ih + 1e-9) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(int(i))
    return rows[kept, :6] if kept else np.zeros((0, 6))


def _make_tracker(tracker_cfg: str):
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_cfg)))
    if getattr(cfg, "with_reid", False):
        raise ValueError("tiled detection does not support ReID trackers; use ocsort or bytetrack in the tracker config")
    return TRACKER_MAP[cfg.tracker_type](args=cfg)

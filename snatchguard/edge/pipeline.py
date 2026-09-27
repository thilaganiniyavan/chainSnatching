"""The per-frame analysis shared by the live camera worker and the offline evaluator.

    image -> lighting -> enhance -> detect+track -> scene/pairs -> pose on closest pairs -> rules
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from snatchguard.core.config import RuleConfig
from snatchguard.core.types import Detection, FrameObservation, Lighting, ObjClass
from snatchguard.edge.lighting import Enhancer, LightingAnalyzer
from snatchguard.rules.engine import Decision, RuleEngine


@dataclass
class FrameResult:
    t: float
    mode: Lighting
    enhanced: np.ndarray
    detections: list[Detection]
    decisions: list[Decision]


class FramePipeline:
    def __init__(self, models: dict, tracker_cfg: str, rules: RuleConfig, lighting: str = "auto",
                 detect_zone: list[float] | None = None) -> None:
        """`detect_zone` = [x1, y1, x2, y2] as fractions of the frame: the detector only looks at this
        crop (at full resolution), which acts as a zoom on the walkway for distant cameras."""
        from snatchguard.edge.perception import Perception  # needs ultralytics
        self.models = models
        self.zone = detect_zone
        self.perception = Perception(models, tracker_cfg)
        self.engine = RuleEngine(rules)
        self.lighting = LightingAnalyzer(forced=None if lighting == "auto" else Lighting(lighting))
        self.enhancer = Enhancer()
        self.max_crops = int(models.get("max_pose_crops", 2))

    def process(self, img: np.ndarray, t: float) -> FrameResult:
        mode = self.lighting.update(img, t)
        enhanced = self.enhancer.apply(img, mode)
        conf = self.models.get("conf", {}).get(mode.value, 0.25)
        h, w = img.shape[:2]
        if self.zone:
            x1, y1 = int(self.zone[0] * w), int(self.zone[1] * h)
            x2, y2 = int(self.zone[2] * w), int(self.zone[3] * h)
            dets = self.perception.detect_track(enhanced[y1:y2, x1:x2], conf)
            off = np.array([x1, y1, x1, y1], dtype=float)
            for d in dets:
                d.box = d.box + off
        else:
            dets = self.perception.detect_track(enhanced, conf)
        self.engine.ingest(FrameObservation(t, (w, h), dets, {}, mode))
        crops = self.engine.pose_requests(self.max_crops)
        if crops:
            persons = {d.track_id: d.box for d in dets if d.cls is ObjClass.PERSON}
            self.engine.attach_poses(self.perception.estimate_poses(enhanced, crops, persons))
        return FrameResult(t, mode, enhanced, dets, self.engine.evaluate_frame())


_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12)]


def draw_debug(img: np.ndarray, engine: RuleEngine, mode: Lighting, fps: float | None = None,
               zone: list[float] | None = None) -> np.ndarray:
    """Overlay what the rules see: pedestrians (green), riders (orange), bikes (red), pairs and poses."""
    out = img.copy()
    if zone:
        h, w = out.shape[:2]
        cv2.rectangle(out, (int(zone[0] * w), int(zone[1] * h)), (int(zone[2] * w), int(zone[3] * h)), (255, 255, 255), 1)
    scene = engine.scene
    for tr in scene.tracks.values():
        if tr.lost_at is not None:
            continue
        x1, y1, x2, y2 = tr.last.box.astype(int)
        if tr.cls.is_vehicle:
            color, label = (40, 40, 230), f"{tr.cls.value[:4]} {tr.track_id}"
        elif tr.track_id in scene.rider_of:
            color, label = (0, 140, 255), f"rider {tr.track_id}"
        elif tr.track_id in scene.established:
            color, label = (60, 200, 60), f"ped {tr.track_id}"
        else:
            color, label = (170, 170, 170), f"new {tr.track_id}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        pose = tr.latest_pose(scene.t)
        if pose is not None:
            for a, b in _SKELETON:
                if pose.conf[a] > 0.3 and pose.conf[b] > 0.3:
                    cv2.line(out, tuple(pose.xy[a].astype(int)), tuple(pose.xy[b].astype(int)), (255, 255, 0), 2)
    for pair in engine.pairs.values():
        v, b = scene.tracks.get(pair.victim_id), scene.tracks.get(pair.bike_id)
        if v is None or b is None or v.lost_at is not None:
            continue
        color = {"WATCH": (200, 200, 200), "CONTACT": (0, 0, 255), "COOLDOWN": (255, 0, 255)}[pair.phase]
        p1 = ((v.last.box[0] + v.last.box[2]) / 2, (v.last.box[1] + v.last.box[3]) / 2)
        p2 = ((b.last.box[0] + b.last.box[2]) / 2, (b.last.box[1] + b.last.box[3]) / 2)
        cv2.line(out, tuple(int(x) for x in p1), tuple(int(x) for x in p2), color, 2 if pair.phase == "CONTACT" else 1)
        if pair.samples:
            s = pair.samples[-1]
            cv2.putText(out, f"{pair.phase} gap={s.gap:.1f}H", (int(p1[0]), int(p1[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    hud = f"{mode.value.upper()}  tracks={sum(1 for t in scene.tracks.values() if t.lost_at is None)}  pairs={len(engine.pairs)}"
    if fps is not None:
        hud += f"  {fps:.1f} fps"
    cv2.rectangle(out, (0, out.shape[0] - 26), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, hud, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out

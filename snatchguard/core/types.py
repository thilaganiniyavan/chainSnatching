"""Plain data types shared by the edge pipeline, the rule engine and the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# COCO-17 keypoint indices used by the rules.
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
WRISTS = (L_WRIST, R_WRIST)


class ObjClass(str, Enum):
    PERSON = "person"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"

    @property
    def is_vehicle(self) -> bool:
        return self is not ObjClass.PERSON


class Lighting(str, Enum):
    DAY = "day"
    LOWLIGHT = "lowlight"
    IR = "ir"


class Level(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUSPECT = "SUSPECT"
    DISMISSED = "DISMISSED"


@dataclass(slots=True)
class Detection:
    """One tracked object in one frame. `box` is (x1, y1, x2, y2) in pixels."""

    track_id: int
    cls: ObjClass
    box: np.ndarray
    conf: float


@dataclass(slots=True)
class Pose:
    """COCO-17 keypoints in frame pixels plus per-keypoint confidence."""

    xy: np.ndarray    # (17, 2)
    conf: np.ndarray  # (17,)

    def point(self, idx: int, min_conf: float) -> np.ndarray | None:
        return self.xy[idx] if self.conf[idx] >= min_conf else None


@dataclass(slots=True)
class FrameObservation:
    """Everything perception produced for one processed frame."""

    t: float                       # seconds on the camera's clock
    frame_size: tuple[int, int]    # (width, height)
    detections: list[Detection]
    poses: dict[int, Pose] = field(default_factory=dict)  # track_id -> pose
    lighting: Lighting = Lighting.DAY

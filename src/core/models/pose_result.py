"""PoseResult domain model for the framework.

Represents 2D/3D skeleton keypoint estimations for a single person track in a given frame.

Carries pixel coordinates, normalized coordinates, keypoint visibilities, keypoint confidences,
overall pose confidence, and pose quality scores.

Model-agnostic: supports 17-keypoint COCO topology or 33-keypoint MediaPipe topology.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseResult:
    """Pose estimation output for a single person track in a frame.

    Attributes:
        sample_id: Identifier of the source PreparedSkeletonSample (e.g. ``SMP-0001``).
        roi_id: Identifier of the source InteractionROI.
        interaction_id: Source interaction ID.
        frame_index: Video frame index.
        timestamp: Timestamp in seconds from video start.
        track_id: Person tracking ID.
        keypoints_pixel: List of keypoint tuples ``(x_px, y_px, confidence, visibility)``.
        keypoints_normalized: List of normalized keypoint tuples ``(x_norm, y_norm, confidence, visibility)`` in [0, 1].
        num_keypoints: Total keypoints count (17 or 33).
        topology: Keypoint topology label (``COCO_17`` or ``MEDIAPIPE_33``).
        overall_confidence: Mean keypoint confidence score in [0, 1].
        quality_score: Combined pose quality score (mean confidence * completeness).
        bbox_reference: Person bounding box reference tuple ``(x1, y1, x2, y2)``.
        backend_name: Name of backend estimator (``MediaPipe``, ``RTMPose``, ``ViTPose``, ``MMPose``, ``OpenPose``).
        processing_time_ms: Inference time in milliseconds for this pose sample.
        metadata: Arbitrary metadata.
    """

    sample_id: str = field(default_factory=lambda: f"SMP-{uuid.uuid4().hex[:8].upper()}")
    roi_id: str = ""
    interaction_id: str = ""
    frame_index: int = 0
    timestamp: float = 0.0
    track_id: int = -1

    keypoints_pixel: list[tuple[float, float, float, float]] = field(default_factory=list)
    keypoints_normalized: list[tuple[float, float, float, float]] = field(default_factory=list)

    num_keypoints: int = 17
    topology: str = "COCO_17"

    overall_confidence: float = 0.0
    quality_score: float = 0.0
    bbox_reference: tuple[int, int, int, int] = (0, 0, 0, 0)
    backend_name: str = "MediaPipe"
    processing_time_ms: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

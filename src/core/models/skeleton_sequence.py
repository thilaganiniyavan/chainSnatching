"""SkeletonSequence domain model.

Represents a temporally ordered sequence of 2D/3D skeleton keypoints for a person track
over an interaction window. Stores normalized tensors of shape (T, V, C) where:
- T = frames count
- V = joints count (17 COCO or 33 MediaPipe)
- C = channels (x, y, confidence, visibility)

Model-agnostic: designed as standardized input for ST-GCN, CTR-GCN, MSG3D, PoseC3D, etc.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SkeletonSequence:
    """Temporally ordered sequence of pose estimations for a single interaction participant.

    Attributes:
        sequence_id: Identifier of sequence (e.g. ``SEQ-INT-0001-TRK-1``).
        interaction_id: Source interaction ID.
        person_track_id: Person track ID.
        start_frame: Start frame index.
        end_frame: End frame index.
        frame_count: Number of frames T in the sequence.
        duration_seconds: Duration in seconds.
        topology: Keypoint topology (``COCO_17`` or ``MEDIAPIPE_33``).
        num_joints: Number of joints V (17 or 33).
        skeleton_tensor: NumPy array of shape ``(T, V, 4)`` containing ``(x, y, conf, vis)``.
        joint_confidence_matrix: NumPy array of shape ``(T, V)``.
        visibility_matrix: NumPy array of shape ``(T, V)``.
        timestamps: Chronological list of frame timestamps.
        frame_indices: Chronological list of frame indices.
        normalization_method: Normalization strategy used (``hip_centered``, ``bbox``, ``root_joint``, ``image``).
        quality_score: Overall sequence quality score [0, 1].
        completeness_score: Sequence frame completeness score [0, 1].
        is_accepted: Boolean acceptance flag based on quality thresholds.
        rejection_reason: Explanation of rejection or "Accepted".
        metadata: Arbitrary metadata dictionary.
    """

    sequence_id: str = field(default_factory=lambda: f"SEQ-{uuid.uuid4().hex[:8].upper()}")
    interaction_id: str = ""
    person_track_id: int = -1

    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0

    topology: str = "COCO_17"
    num_joints: int = 17

    skeleton_tensor: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 17, 4), dtype=float)
    )
    joint_confidence_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 17), dtype=float)
    )
    visibility_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 17), dtype=float)
    )

    timestamps: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    normalization_method: str = "hip_centered"
    quality_score: float = 0.0
    completeness_score: float = 0.0

    is_accepted: bool = False
    rejection_reason: str = "Initialized"

    metadata: dict[str, Any] = field(default_factory=dict)

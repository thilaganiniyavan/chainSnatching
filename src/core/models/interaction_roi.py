"""Interaction ROI and Prepared Skeleton Sample domain models.

An InteractionROI represents a spatial-temporal window surrounding a person track
during an active or completed interaction selected by the Behaviour Graph Engine.

A PreparedSkeletonSample represents a standardized per-frame input sample prepared
for future pose estimation modules (MediaPipe, RTMPose, ViTPose, MMPose, OpenPose).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InteractionROI:
    """Represents a cropped spatial-temporal interaction window selected for pose analysis.

    Attributes:
        roi_id: Unique identifier for this ROI (e.g. ``ROI-INT-0001``).
        interaction_id: Identifier of the source interaction.
        video_id: Source video identifier.
        start_frame: First frame of the interaction window.
        end_frame: Last frame of the interaction window.
        frame_count: Total number of frames in the window.
        duration_seconds: Duration converted via video FPS.
        person_track_id: Tracking ID of the person participant.
        vehicle_track_id: Tracking ID of the vehicle participant.
        bounding_box_sequence: Sequence of smoothed person bounding boxes ``[x1, y1, x2, y2]``.
        expanded_bounding_boxes: Sequence of context-expanded bounding boxes ``[x1, y1, x2, y2]``.
        frame_index_mapping: List of video frame numbers corresponding 1-to-1 with sequence indices.
        timestamps: List of seconds from video start for each frame.
        graph_reference_id: Graph ID of the source BehaviourGraph.
        interaction_confidence: Graph/interaction confidence score in [0, 1].
        pattern_sequence: Ordered list of Behaviour Patterns observed during this window.
        quality_metrics: Quantitative quality scores (completeness, missing_pct, stability, continuity, coverage).
        is_accepted: Whether this ROI passed quality threshold checks.
        rejection_reason: Human-readable explanation if rejected.
        metadata: Extensibility hook for future ST-GCN / Fusion metadata.
    """

    roi_id: str = field(default_factory=lambda: f"ROI-{uuid.uuid4().hex[:8].upper()}")
    interaction_id: str = ""
    video_id: str = "video_001"
    start_frame: int = 0
    end_frame: int = 0
    frame_count: int = 0
    duration_seconds: float = 0.0

    person_track_id: int = -1
    vehicle_track_id: int = -1

    bounding_box_sequence: list[tuple[int, int, int, int]] = field(default_factory=list)
    expanded_bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    frame_index_mapping: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    graph_reference_id: str = ""
    interaction_confidence: float = 0.0
    pattern_sequence: list[str] = field(default_factory=list)

    quality_metrics: dict[str, float] = field(default_factory=dict)
    is_accepted: bool = False
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedSkeletonSample:
    """Standardized input sample prepared for downstream pose estimators.

    Compatible with MediaPipe, RTMPose, ViTPose, MMPose, and OpenPose interfaces.

    Attributes:
        sample_id: Unique sample identifier (e.g. ``SAMPLE-0001``).
        roi_id: Source InteractionROI identifier.
        interaction_id: Source interaction identifier.
        frame_number: Video frame number.
        timestamp: Seconds from video start.
        person_track_id: Person tracking ID.
        raw_bbox: Raw person bounding box ``(x1, y1, x2, y2)``.
        expanded_bbox: Expanded context bounding box ``(x1, y1, x2, y2)``.
        expected_skeleton_placeholder: Dict template ready for keypoints/scores output.
        metadata: Extensibility metadata.
    """

    sample_id: str = field(default_factory=lambda: f"SMP-{uuid.uuid4().hex[:8].upper()}")
    roi_id: str = ""
    interaction_id: str = ""
    frame_number: int = 0
    timestamp: float = 0.0
    person_track_id: int = -1
    raw_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    expanded_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    expected_skeleton_placeholder: dict[str, Any] = field(
        default_factory=lambda: {
            "topology": "COCO_17",
            "num_keypoints": 17,
            "keypoints_2d": None,      # Array of shape [17, 3] (x, y, confidence)
            "pose_score": 0.0,
            "model_compatibility": ["MediaPipe", "RTMPose", "ViTPose", "MMPose", "OpenPose"],
        }
    )
    metadata: dict[str, Any] = field(default_factory=list)

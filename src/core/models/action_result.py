"""ActionResult domain model for the Human Action Recognition Layer.

Represents physical human action classifications (e.g. Walking, Standing, Running,
Approaching, Reaching, Grabbing, Pulling, Turning, Falling, Unknown) extracted from
SkeletonSequence input tensors.

Model-agnostic: supports ST-GCN, CTR-GCN, MSG-3D, PoseC3D, and future skeleton GNN models.
Does NOT perform chain-snatching classification or behaviour fusion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    """Action classification result for a skeleton sequence.

    Attributes:
        action_id: Identifier of action result (e.g. ``ACT-0001``).
        sequence_id: Source SkeletonSequence ID.
        interaction_id: Source interaction ID.
        track_id: Person track ID.
        predicted_action: Primary predicted action label.
        action_confidence: Confidence score of primary prediction in [0.0, 1.0].
        class_probabilities: Mapping from action class name to probability score.
        top_k_predictions: Ranked list of ``(action_class, probability)`` tuples.
        inference_time_ms: Inference execution latency in milliseconds.
        model_name: Name of recognizer backend (``ST-GCN``, ``CTR-GCN``, ``MSG-3D``, ``PoseC3D``).
        model_version: Version string of model backend.
        device_used: Execution hardware device (``CUDA:0`` or ``CPU``).
        skeleton_quality: Quality score of input SkeletonSequence.
        metadata: Arbitrary metadata.
    """

    action_id: str = field(default_factory=lambda: f"ACT-{uuid.uuid4().hex[:8].upper()}")
    sequence_id: str = ""
    interaction_id: str = ""
    track_id: int = -1

    predicted_action: str = "Unknown"
    action_confidence: float = 0.0
    class_probabilities: dict[str, float] = field(default_factory=dict)
    top_k_predictions: list[tuple[str, float]] = field(default_factory=list)

    inference_time_ms: float = 0.0
    model_name: str = "ST-GCN"
    model_version: str = "1.0.0"
    device_used: str = "CPU"
    skeleton_quality: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

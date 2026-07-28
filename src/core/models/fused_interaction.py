"""FusedInteraction domain model.

Represents a multi-modal fused interaction combining Stream A (Behaviour Graph patterns,
spatial/motion evidence) and Stream B (Human Action Recognition predictions, ST-GCN confidence,
action timelines).

Carries behaviour confidence, action confidence, overall fusion confidence score,
multi-modal evidence timeline, and human-readable explanation text.

Does NOT classify chain-snatching or infer suspicious crimes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FusedInteraction:
    """Multi-modal fused interaction representation.

    Attributes:
        fusion_id: Identifier of fused interaction (e.g. ``FUSED-INT-0001``).
        interaction_id: Source interaction ID.
        person_track_id: Person participant track ID.
        vehicle_track_id: Vehicle participant track ID (-1 if person-person).
        start_frame: Interaction start frame index.
        end_frame: Interaction end frame index.
        duration_seconds: Interaction duration in seconds.
        behaviour_patterns: List of detected Behaviour Graph pattern labels.
        action_timeline: List of chronological action prediction dictionaries.
        motion_evidence: Dictionary of motion statistics (speed, acceleration, trajectory).
        spatial_evidence: Dictionary of spatial relationship statistics (min distance, proximity).
        temporal_evidence: Dictionary of temporal span statistics.
        action_evidence: List of supporting action classification predictions.
        behaviour_confidence: Confidence score of Behaviour Graph stream [0, 1].
        action_confidence: Confidence score of Action Recognition stream [0, 1].
        fusion_confidence: Combined multi-modal fusion confidence score [0, 1].
        fusion_strategy: Name of fusion strategy used (``weighted_confidence``, ``bayesian``, etc.).
        evidence_timeline: Chronological list of synchronized multi-modal evidence events.
        explanation_text: Human-readable explainable text summary backed by evidence.
        metadata: Arbitrary metadata.
    """

    fusion_id: str = field(default_factory=lambda: f"FUSED-{uuid.uuid4().hex[:8].upper()}")
    interaction_id: str = ""
    person_track_id: int = -1
    vehicle_track_id: int = -1

    start_frame: int = 0
    end_frame: int = 0
    duration_seconds: float = 0.0

    behaviour_patterns: list[str] = field(default_factory=list)
    action_timeline: list[dict[str, Any]] = field(default_factory=list)

    motion_evidence: dict[str, Any] = field(default_factory=dict)
    spatial_evidence: dict[str, Any] = field(default_factory=dict)
    temporal_evidence: dict[str, Any] = field(default_factory=dict)
    action_evidence: list[dict[str, Any]] = field(default_factory=list)

    behaviour_confidence: float = 0.0
    action_confidence: float = 0.0
    fusion_confidence: float = 0.0
    fusion_strategy: str = "weighted_confidence"

    evidence_timeline: list[dict[str, Any]] = field(default_factory=list)
    explanation_text: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

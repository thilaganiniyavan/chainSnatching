"""Behaviour Event domain model for the Reasoning Engine.

A BehaviourEvent represents a higher-level classification produced by
composing sequences of behavioural primitives.  Each event carries
the full evidence chain (supporting primitives, motion measurements,
spatial measurements) and a human-readable explanation so that every
classification decision is explainable and forensically auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehaviourEvent:
    """A higher-level behavioural classification derived from primitive sequences.

    Attributes:
        event_id: Unique identifier (e.g. ``EVT-0001``).
        event_type: Classification label such as ``CLOSE_ENCOUNTER``.
        confidence: Classification confidence in [0, 1].
        start_frame: First video frame of the event.
        end_frame: Last video frame of the event.
        duration_frames: ``end_frame - start_frame + 1``.
        duration_seconds: Duration converted via FPS.
        participants: ``{person_track_id, vehicle_track_id}``.
        interaction_id: The source interaction this event was derived from.
        supporting_sequence: Ordered list of primitive types that triggered
            this classification.
        motion_evidence: Quantitative kinematic measurements supporting the
            classification (avg/max speed, acceleration, etc.).
        spatial_evidence: Quantitative spatial measurements (min/avg/max
            distance, final distance, etc.).
        explanation: Human-readable reasoning sentence.
        is_tentative: ``True`` when derived from an active (incomplete)
            interaction — lower reliability.
        metadata: Arbitrary additional data.
    """

    event_id: str = ""
    event_type: str = ""
    confidence: float = 0.0

    start_frame: int = 0
    end_frame: int = 0
    duration_frames: int = 0
    duration_seconds: float = 0.0

    participants: dict[str, int] = field(default_factory=dict)
    interaction_id: str = ""

    supporting_sequence: list[str] = field(default_factory=list)
    motion_evidence: dict[str, Any] = field(default_factory=dict)
    spatial_evidence: dict[str, Any] = field(default_factory=dict)

    explanation: str = ""
    is_tentative: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

"""Interaction domain model for person-vehicle encounter tracking.

An Interaction Object represents a persistent, stateful relationship between
a person track and a vehicle track across video frames.  It captures spatial,
kinematic, and temporal measurements needed by the Behaviour Intelligence
Engine to reason about the nature of the encounter.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InteractionState(Enum):
    """Lifecycle states for a tracked interaction.

    Transitions:
        NEW -> ACTIVE -> LINGERING -> ENDED -> ARCHIVED
    """

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LINGERING = "LINGERING"
    ENDED = "ENDED"
    ARCHIVED = "ARCHIVED"


@dataclass
class Interaction:
    """Represents an ongoing or completed interaction between a person and a vehicle.

    All distance values are in pixel-space.
    All velocity / acceleration values are in px/frame units.
    """

    # ---- Identity ----
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    person_track_id: int = -1
    vehicle_track_id: int = -1

    # ---- Temporal ----
    start_frame: int = 0
    current_frame: int = 0
    end_frame: int | None = None
    duration: int = 0

    # ---- Spatial statistics ----
    min_distance: float = float("inf")
    max_distance: float = 0.0
    avg_distance: float = 0.0
    current_distance: float = 0.0

    # ---- Kinematics ----
    relative_velocity: float = 0.0        # +ve = separating, -ve = closing
    relative_acceleration: float = 0.0
    heading_difference: float = 0.0       # degrees
    trajectory_similarity: float = 0.0    # cosine similarity [-1, 1]

    # ---- Confidence ----
    interaction_confidence: float = 0.0

    # ---- Lifecycle ----
    state: InteractionState = InteractionState.NEW

    # ---- History ----
    relationship_history: list[dict[str, Any]] = field(default_factory=list)
    motion_history: list[dict[str, Any]] = field(default_factory=list)

    # ---- Internal bookkeeping ----
    _distance_sum: float = field(default=0.0, repr=False)
    _distance_count: int = field(default=0, repr=False)
    _previous_velocity: float | None = field(default=None, repr=False)
    _frames_since_last_seen: int = field(default=0, repr=False)

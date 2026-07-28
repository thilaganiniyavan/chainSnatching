"""Pattern Rules Configuration for the Behaviour Graph Engine.

Defines configurable thresholds and rule parameters for evaluating the 11
reusable Behaviour Patterns:

- APPROACH_PATTERN
- FOLLOW_PATTERN
- CO_TRAVEL_PATTERN
- PROXIMITY_PATTERN
- INTERACTION_PATTERN
- STOP_PATTERN
- LINGERING_PATTERN
- SEPARATION_PATTERN
- ESCAPE_PATTERN
- DIVERGENCE_PATTERN
- WAITING_PATTERN

All parameters are constructor-injected so thresholds can be tuned for
ablation studies without changing core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatternConfig:
    """Configurable threshold parameters for pattern detection.

    All distance thresholds are in pixels.
    All velocity / acceleration thresholds are in px/frame.
    """

    # Spatial thresholds
    proximity_distance_threshold: float = 80.0
    close_interaction_distance_threshold: float = 60.0

    # Kinematic thresholds
    approach_velocity_threshold: float = -1.5
    separation_velocity_threshold: float = 2.0
    escape_acceleration_threshold: float = 2.5
    escape_velocity_threshold: float = 4.0
    stationary_speed_threshold: float = 2.0

    # Alignment thresholds
    heading_parallel_threshold: float = 20.0       # degrees
    heading_divergence_threshold: float = 45.0     # degrees
    trajectory_similarity_threshold: float = 0.65  # cosine similarity

    # Duration thresholds (frames)
    min_pattern_frames: int = 3
    min_follow_frames: int = 5
    min_lingering_frames: int = 10
    min_waiting_frames: int = 15

    # Base confidence scores
    base_confidence: float = 0.6

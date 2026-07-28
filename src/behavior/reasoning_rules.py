"""Reasoning Rule Graph — configurable event classification rules.

Each :class:`RuleNode` defines the evidence requirements for one Behaviour
Event type.  Rules specify which behavioural primitives must be present,
optional sequence ordering, temporal/spatial/kinematic constraints, and a
base confidence score.

The default rule set covers seven event types.  Custom rules can be added
by constructing additional ``RuleNode`` instances — no code changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuleNode:
    """A single classification rule mapping evidence to an event type.

    Attributes:
        event_type: The Behaviour Event label produced when this rule fires.
        required_primitives: Primitive types that must appear in the timeline.
            At least ``min_primitive_count`` of these must be present.
        min_primitive_count: Minimum number of *distinct* required primitives
            that must match.
        required_sequence: If set, the matched primitives must appear in this
            order in the timeline (allows gaps).
        min_duration_frames: Minimum interaction duration (frames).
        max_duration_frames: Maximum interaction duration (frames), or None.
        min_distance: Minimum closest-approach distance (px), or None.
        max_distance: Maximum closest-approach distance (px), or None.
        min_velocity: Minimum peak |relative_velocity| (px/frame), or None.
        max_velocity: Maximum peak |relative_velocity| (px/frame), or None.
        min_acceleration: Minimum peak |relative_acceleration|, or None.
        max_acceleration: Maximum peak |relative_acceleration|, or None.
        base_confidence: Starting confidence before evidence scaling.
        priority: Higher priority wins on conflict when multiple rules fire.
        description: Human-readable explanation of what this rule detects.
    """

    event_type: str = ""
    required_primitives: list[str] = field(default_factory=list)
    min_primitive_count: int = 1
    required_sequence: list[str] | None = None

    # Temporal constraints
    min_duration_frames: int = 1
    max_duration_frames: int | None = None

    # Spatial constraints
    min_distance: float | None = None
    max_distance: float | None = None

    # Kinematic constraints
    min_velocity: float | None = None
    max_velocity: float | None = None
    min_acceleration: float | None = None
    max_acceleration: float | None = None

    # Scoring
    base_confidence: float = 0.5
    priority: int = 0

    # Documentation
    description: str = ""


# ======================================================================
# Default rule configuration
# ======================================================================

def get_default_rules() -> list[RuleNode]:
    """Return the default set of seven event classification rules.

    All thresholds use sensible defaults for pixel-space CCTV analysis.
    Override by constructing custom ``RuleNode`` lists.
    """
    return [
        # ---- NORMAL_PASSING ----
        RuleNode(
            event_type="NORMAL_PASSING",
            required_primitives=["APPROACHING", "MOVING_AWAY"],
            min_primitive_count=2,
            required_sequence=["APPROACHING", "MOVING_AWAY"],
            min_duration_frames=2,
            max_duration_frames=90,       # Short interactions only
            max_distance=None,            # No distance constraint
            base_confidence=0.4,
            priority=0,                   # Lowest — easily overridden
            description=(
                "Vehicle passes by a pedestrian without stopping. "
                "Characterised by an approach phase followed by departure, "
                "with no sustained close interaction."
            ),
        ),

        # ---- VEHICLE_WAITING ----
        RuleNode(
            event_type="VEHICLE_WAITING",
            required_primitives=["STATIONARY_INTERACTION"],
            min_primitive_count=1,
            min_duration_frames=30,       # At least ~1 second at 30 fps
            max_velocity=3.0,             # Vehicle must be near-stationary
            base_confidence=0.5,
            priority=1,
            description=(
                "Vehicle idles in proximity to a pedestrian. "
                "Low vehicle speed sustained over a minimum duration."
            ),
        ),

        # ---- FOLLOWING_BEHAVIOUR ----
        RuleNode(
            event_type="FOLLOWING_BEHAVIOUR",
            required_primitives=["FOLLOWING", "PARALLEL_MOTION"],
            min_primitive_count=1,        # Either suffices
            min_duration_frames=15,       # At least ~0.5 second
            base_confidence=0.55,
            priority=2,
            description=(
                "Sustained following pattern where the vehicle tracks "
                "the pedestrian's trajectory with high similarity."
            ),
        ),

        # ---- STATIONARY_INTERACTION ----
        RuleNode(
            event_type="STATIONARY_INTERACTION",
            required_primitives=["STATIONARY_INTERACTION", "CLOSE_INTERACTION"],
            min_primitive_count=2,
            min_duration_frames=15,
            max_distance=80.0,
            max_velocity=2.0,
            base_confidence=0.5,
            priority=4,
            description=(
                "Both objects are near-stationary and within close range. "
                "May indicate conversation, transaction, or waiting."
            ),
        ),

        # ---- CLOSE_ENCOUNTER ----
        RuleNode(
            event_type="CLOSE_ENCOUNTER",
            required_primitives=["CLOSE_INTERACTION", "APPROACHING"],
            min_primitive_count=1,        # At least close interaction
            max_distance=60.0,            # Must get very close
            min_duration_frames=5,
            base_confidence=0.55,
            priority=3,
            description=(
                "Significant proximity event where the vehicle and "
                "pedestrian come within a tight distance threshold."
            ),
        ),

        # ---- SUSPICIOUS_ENCOUNTER ----
        RuleNode(
            event_type="SUSPICIOUS_ENCOUNTER",
            required_primitives=[
                "APPROACHING",
                "CLOSE_INTERACTION",
                "RAPID_ACCELERATION",
            ],
            min_primitive_count=2,        # At least 2 of the 3
            required_sequence=["APPROACHING", "CLOSE_INTERACTION"],
            min_duration_frames=5,
            max_distance=80.0,
            min_acceleration=2.0,
            base_confidence=0.6,
            priority=5,                   # High priority
            description=(
                "Approach followed by close interaction and sudden "
                "acceleration. The approach-contact-accelerate pattern "
                "indicates potentially concerning behaviour."
            ),
        ),

        # ---- RAPID_ESCAPE ----
        RuleNode(
            event_type="RAPID_ESCAPE",
            required_primitives=[
                "RAPID_ACCELERATION",
                "RAPID_SEPARATION",
            ],
            min_primitive_count=2,
            required_sequence=["RAPID_ACCELERATION", "RAPID_SEPARATION"],
            min_acceleration=2.5,
            min_velocity=4.0,
            base_confidence=0.6,
            priority=6,                   # Highest
            description=(
                "Sudden departure after an interaction. High acceleration "
                "combined with rapid increase in distance."
            ),
        ),
    ]

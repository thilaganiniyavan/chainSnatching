"""Signature Configuration & Evidence Templates.

Defines configurable signature templates for chain-snatching events:
- ``StandardMotorcycleSnatchSignature`` (motorcycle vehicle rider + pedestrian victim)
- ``PedestrianSnatchSignature`` (pedestrian-on-pedestrian snatching)

Implements the 4-State Weighted Chronological State Model:
- S0: Closing Approach Vector (0.20)
- S1: Directed Physical Grab / Attack (0.35)
- S2: Victim Reactive Jerk / Deflection (0.25)
- S3: Rapid Getaway / Escape (0.20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignatureTemplate:
    """Configurable template defining evidence requirements for a forensic signature.

    Attributes:
        signature_name: Unique identifier of template (e.g. ``StandardMotorcycleSnatch``).
        description: Human-readable template description.
        required_patterns: Behaviour Graph pattern names required.
        optional_patterns: Optional Behaviour Graph pattern names.
        target_actions: Pose-based action labels considered physical attack relevant.
        min_duration_seconds: Minimum interaction temporal duration threshold.
        max_proximity_px: Maximum proximity distance threshold.
        min_average_speed: Minimum relative speed threshold.
        evidence_weights: Dictionary mapping evidence component names to weights in [0, 1].
        decision_thresholds: Dictionary mapping decision labels to minimum score boundaries.
    """

    signature_name: str = "StandardMotorcycleSnatch"
    description: str = "Motorcycle-borne snatcher approaching, reaching/grabbing, and escaping."

    required_patterns: list[str] = field(
        default_factory=lambda: ["APPROACH_PATTERN", "INTERACTION_PATTERN"]
    )
    optional_patterns: list[str] = field(
        default_factory=lambda: [
            "REACH_GRAB_RETRACT_PATTERN",
            "ESCAPE_PATTERN",
            "FOLLOW_PATTERN",
            "PROXIMITY_PATTERN",
            "DIVERGENCE_PATTERN",
        ]
    )
    target_actions: list[str] = field(
        default_factory=lambda: ["Grabbing", "Reaching", "Pulling", "Falling"]
    )

    min_duration_seconds: float = 0.50
    max_proximity_px: float = 140.0
    min_average_speed: float = 3.0

    evidence_weights: dict[str, float] = field(
        default_factory=lambda: {
            "closing_approach_vector": 0.20,    # S0: Closing Approach Vector
            "directed_grab": 0.35,              # S1: Directed Grab / Physical Attack
            "victim_reactive_jerk": 0.25,       # S2: Victim Reactive Jerk / Deflection
            "rapid_getaway_escape": 0.20,       # S3: Rapid Getaway / Escape
        }
    )

    decision_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "High Confidence Match": 0.80,
            "Strong Match": 0.65,
            "Partial Match": 0.45,
            "Weak Match": 0.20,
            "No Match": 0.0,
        }
    )


class StandardMotorcycleSnatchSignature(SignatureTemplate):
    """Standard template for motorcycle-borne chain snatching."""

    def __init__(self) -> None:
        super().__init__(
            signature_name="StandardMotorcycleSnatch",
            description="Motorcycle rider approaches pedestrian victim, reaches/grabs object, and escapes rapidly.",
        )


class PedestrianSnatchSignature(SignatureTemplate):
    """Alternative template for pedestrian-on-pedestrian chain snatching."""

    def __init__(self) -> None:
        super().__init__(
            signature_name="PedestrianSnatch",
            description="Pedestrian approaches victim, reaches/grabs object, and flees on foot.",
            required_patterns=["APPROACH_PATTERN", "INTERACTION_PATTERN"],
            optional_patterns=["ESCAPE_PATTERN", "SEPARATION_PATTERN"],
            target_actions=["Grabbing", "Reaching", "Running"],
        )

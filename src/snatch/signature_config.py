"""Signature Configuration & Evidence Templates.

Defines configurable signature templates for chain-snatching events:
- ``StandardMotorcycleSnatchSignature`` (motorcycle vehicle rider + pedestrian victim)
- ``PedestrianSnatchSignature`` (pedestrian-on-pedestrian snatching)

Allows customizing required/optional pattern components, action targets,
kinematic thresholds, evidence weights, and decision classification boundaries.
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
        required_patterns: Behaviour Graph pattern names required (e.g. ``["APPROACH_PATTERN", "INTERACTION_PATTERN"]``).
        optional_patterns: Optional Behaviour Graph pattern names (e.g. ``["ESCAPE_PATTERN", "FOLLOW_PATTERN"]``).
        target_actions: Pose-based action labels considered relevant (e.g. ``["Reaching", "Grabbing", "Pulling"]``).
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
        default_factory=lambda: ["ESCAPE_PATTERN", "FOLLOW_PATTERN", "PROXIMITY_PATTERN"]
    )
    target_actions: list[str] = field(
        default_factory=lambda: ["Reaching", "Grabbing", "Pulling", "Running"]
    )

    min_duration_seconds: float = 0.50
    max_proximity_px: float = 150.0
    min_average_speed: float = 0.05

    evidence_weights: dict[str, float] = field(
        default_factory=lambda: {
            "approach_pattern": 0.15,
            "interaction_pattern": 0.20,
            "target_action": 0.25,
            "rapid_acceleration": 0.15,
            "escape_pattern": 0.15,
            "proximity_constraint": 0.10,
        }
    )

    decision_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "High Confidence Match": 0.85,
            "Strong Match": 0.70,
            "Partial Match": 0.55,
            "Weak Match": 0.35,
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
            target_actions=["Reaching", "Grabbing", "Running"],
        )

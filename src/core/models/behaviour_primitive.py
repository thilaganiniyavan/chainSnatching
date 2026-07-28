"""Behaviour primitive domain model for the Behaviour Intelligence Engine.

A BehaviourPrimitive represents a single, independently detectable behavioural
signal extracted from an Interaction Object.  Each primitive carries the frame
range over which it was observed, a confidence score, and the raw measurements
that support the classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehaviourPrimitive:
    """A single behavioural signal detected within an interaction.

    Attributes:
        primitive_type: Identifier such as ``APPROACHING``, ``RAPID_SEPARATION``.
        interaction_id: The interaction from which this primitive was derived.
        start_frame: First frame where the behaviour was observed.
        end_frame: Last frame where the behaviour was observed.
        confidence: Detection confidence in [0, 1].
        measurements: Supporting quantitative evidence.
    """

    primitive_type: str
    interaction_id: str
    start_frame: int
    end_frame: int
    confidence: float = 0.0
    measurements: dict[str, Any] = field(default_factory=dict)

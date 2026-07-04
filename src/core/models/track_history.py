"""Track history data model for the surveillance framework.

This module defines the stored history associated with a tracked object.
It contains only data fields and no tracking logic.
"""

from dataclasses import dataclass, field


@dataclass
class TrackHistory:
    """Stores historical movement information for a tracked object."""

    tracking_id: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    directions: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
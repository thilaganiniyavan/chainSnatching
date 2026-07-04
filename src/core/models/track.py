"""Track domain model placeholder for future tracking stages."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Track:
    """Placeholder model representing a tracked object."""

    tracking_id: int
    class_name: str
    detections: list[Any] = field(default_factory=list)
    trajectory: list[Any] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    center: tuple[int, int] | None = None
    history: Any = None
    instantaneous_speed: float | None = None
    average_speed: float | None = None
    direction: float | None = None
    distance_travelled: float | None = None
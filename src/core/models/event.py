"""Event domain model placeholder for future behavioral analysis stages."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Placeholder model representing a detected event."""

    event_id: int
    event_type: str
    timestamp: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
"""Relationship domain model for associations between objects."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Relationship:
    """Represents a relationship between two tracked or detected objects."""

    subject_id: int
    subject_class: str
    object_id: int
    object_class: str
    relationship_type: str
    distance: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
"""Detection domain model for the surveillance framework."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """Represents one detected object."""

    class_id: int
    class_name: str

    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    center_x: int
    center_y: int

    width: int
    height: int

    area: int
    tracking_id: Optional[int] = None

    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2
        )

    def center(self) -> Tuple[int, int]:
        return (
            self.center_x,
            self.center_y
        )
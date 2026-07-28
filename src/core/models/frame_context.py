"""Frame context data structures for the surveillance pipeline.

This module defines the shared state object that moves through each stage of
the processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameContext:
    """Container for per-frame data passed between pipeline stages."""

    frame: Any
    frame_number: int
    timestamp: float
    detections: list[Any] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    depth_map: Any | None = None
    poses: list[Any] = field(default_factory=list)
    interactions: list[Any] = field(default_factory=list)
    graphs: list[Any] = field(default_factory=list)
    rois: list[Any] = field(default_factory=list)
    sequences: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    fused_interactions: list[Any] = field(default_factory=list)
    snatch_signatures: list[Any] = field(default_factory=list)
    forensic_events: list[Any] = field(default_factory=list)
    behavior: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
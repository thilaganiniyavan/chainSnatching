"""Behaviour Graph domain models for the framework.

A Behaviour Graph represents the temporal and structural evolution of a tracked
interaction as a directed state-transition graph.

Nodes represent higher-level Behaviour Patterns (composed from sequences of
primitive behaviours, kinematics, and spatial statistics).

Edges represent directed temporal transitions between consecutive patterns.

Designed with future extensibility in mind so skeleton pose actions,
ST-GCN features, and behaviour fusion embeddings can be attached to nodes
and graph metadata without changing the core graph structure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatternNode:
    """A node in a Behaviour Graph representing an observed Behaviour Pattern.

    Attributes:
        pattern_id: Unique identifier for this pattern instance (e.g. ``PAT-0001``).
        pattern_type: Pattern category label (e.g. ``APPROACH_PATTERN``).
        start_frame: Frame where this pattern started.
        end_frame: Frame where this pattern ended or was updated.
        duration_frames: Number of frames elapsed in this pattern.
        duration_seconds: Duration converted via FPS.
        confidence: Classification confidence score in [0, 1].
        supporting_primitives: List of primitive behaviour types contributing to this pattern.
        supporting_motion: Motion and kinematic measurements (speeds, velocities, accelerations).
        supporting_spatial: Spatial measurements (min/avg/max distances).
        extensible_evidence: Hook for future Pose / ST-GCN / Fusion evidence.
    """

    pattern_id: str = field(default_factory=lambda: f"PAT-{uuid.uuid4().hex[:8].upper()}")
    pattern_type: str = ""
    start_frame: int = 0
    end_frame: int = 0
    duration_frames: int = 0
    duration_seconds: float = 0.0
    confidence: float = 0.0
    priority: int = 0

    supporting_primitives: list[str] = field(default_factory=list)
    supporting_motion: dict[str, Any] = field(default_factory=dict)
    supporting_spatial: dict[str, Any] = field(default_factory=dict)
    extensible_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionEdge:
    """A directed edge in a Behaviour Graph representing a temporal transition between patterns.

    Attributes:
        source_pattern_id: Pattern ID of the source node.
        target_pattern_id: Pattern ID of the target node.
        from_pattern_type: Pattern type of the source node.
        to_pattern_type: Pattern type of the target node.
        transition_frame: Frame number when the transition occurred.
        timestamp: Seconds from video start when transition occurred.
        transition_confidence: Confidence score of the transition.
        transition_condition: Description of the evidence justifying the transition.
    """

    source_pattern_id: str
    target_pattern_id: str
    from_pattern_type: str
    to_pattern_type: str
    transition_frame: int
    timestamp: float
    transition_confidence: float = 1.0
    transition_condition: str = "sequential_transition"


@dataclass
class BehaviourGraph:
    """A directed graph representing the temporal evolution of a single interaction.

    Attributes:
        graph_id: Unique graph identifier (e.g. ``GRAPH-INT-0001``).
        interaction_id: Identifier of the tracked interaction.
        person_track_id: Person tracking ID.
        vehicle_track_id: Vehicle tracking ID.
        start_frame: Frame number where the graph was created.
        end_frame: Frame number where the graph was finalized, or None if active.
        is_active: Whether the graph is currently receiving updates.
        nodes: Directed list of PatternNode instances.
        edges: Directed list of TransitionEdge instances.
        extensible_metadata: Metadata hook for graph-level embeddings or pose fusion.
    """

    graph_id: str = ""
    interaction_id: str = ""
    person_track_id: int = -1
    vehicle_track_id: int = -1
    start_frame: int = 0
    end_frame: int | None = None
    is_active: bool = True

    nodes: list[PatternNode] = field(default_factory=list)
    edges: list[TransitionEdge] = field(default_factory=list)
    extensible_metadata: dict[str, Any] = field(default_factory=dict)

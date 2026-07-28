"""ForensicEvent domain model.

Represents a searchable forensic event record linking all multi-stage evidence artifacts:
- Behaviour Graph reference
- Action Timeline reference
- ROI reference
- Pose estimation reference
- Skeleton tensor reference
- Multi-modal fusion reference
- Thumbnail image path & preview video clip path
- Investigator notes and search tags

Organises evidence for query indexing without modifying upstream detection results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ForensicEvent:
    """Searchable forensic event record.

    Attributes:
        event_id: Unique forensic event identifier (e.g. ``EVT-0001``).
        video_id: Source video identifier or file name.
        interaction_id: Source interaction ID.
        fusion_id: Source multi-modal fusion ID.
        signature_id: Source snatch signature ID.
        timestamp: Event occurrence timestamp.
        start_frame: Interaction start frame index.
        end_frame: Interaction end frame index.
        duration_seconds: Event duration in seconds.
        person_track_id: Person participant track ID.
        vehicle_track_id: Vehicle participant track ID (-1 if person-person).
        location: Camera or CCTV location identifier string.
        decision: Crime matching decision (``No Match``, ``Weak Match``, ``Partial Match``, ``Strong Match``, ``High Confidence Match``).
        signature_score: Weighted signature match score in [0.0, 1.0].
        confidence: Overall forensic confidence score in [0.0, 1.0].
        matched_signature_name: Name of matched signature template.
        behaviour_patterns: List of observed Behaviour Graph pattern names.
        detected_actions: List of detected pose action labels.
        evidence_summary: Dictionary summarizing matched vs missing evidence count.
        behaviour_graph_ref: Reference path or ID to source Behaviour Graph.
        action_timeline_ref: Reference path or ID to source Action Timeline.
        roi_ref: Reference path or ID to source Interaction ROI.
        pose_ref: Reference path or ID to source PoseResult.
        skeleton_ref: Reference path or ID to source SkeletonSequence.
        fusion_ref: Reference path or ID to source FusedInteraction.
        thumbnail_path: File path to keyframe thumbnail image.
        video_clip_path: File path to annotated event preview video clip.
        investigator_notes: Optional investigator text notes.
        tags: List of searchable keywords/tags.
        metadata: Arbitrary metadata.
    """

    event_id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:8].upper()}")
    video_id: str = "default_video"
    interaction_id: str = ""
    fusion_id: str = ""
    signature_id: str = ""

    timestamp: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    duration_seconds: float = 0.0

    person_track_id: int = -1
    vehicle_track_id: int = -1
    location: str = "Camera 1"

    decision: str = "No Match"
    signature_score: float = 0.0
    confidence: float = 0.0
    matched_signature_name: str = "StandardMotorcycleSnatch"

    behaviour_patterns: list[str] = field(default_factory=list)
    detected_actions: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    behaviour_graph_ref: str = ""
    action_timeline_ref: str = ""
    roi_ref: str = ""
    pose_ref: str = ""
    skeleton_ref: str = ""
    fusion_ref: str = ""

    thumbnail_path: str = ""
    video_clip_path: str = ""
    investigator_notes: str = ""
    tags: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

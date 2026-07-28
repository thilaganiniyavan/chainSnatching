"""SnatchSignatureResult domain model.

Represents crime-specific forensic evaluation results for chain-snatching signatures.
Stores matched vs missing evidence items, weighted signature score, decision label
(No Match, Weak Match, Partial Match, Strong Match, High Confidence Match), confidence,
human-readable explanation text with evidence checkmarks, and investigator recommendations.

Designed for direct indexability by the future Forensic Indexing Engine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SnatchSignatureResult:
    """Forensic snatch signature evaluation result.

    Attributes:
        signature_id: Identifier of signature result (e.g. ``SIG-0001``).
        interaction_id: Source interaction ID.
        fusion_id: Source multi-modal fusion ID.
        matched_signature_name: Name of evaluated signature template (e.g. ``MotorcycleChainSnatch``).
        signature_score: Weighted signature match score in [0.0, 1.0].
        decision: Crime matching decision label (``No Match``, ``Weak Match``, ``Partial Match``, ``Strong Match``, ``High Confidence Match``).
        confidence: Overall evaluation confidence score in [0.0, 1.0].
        matched_evidence: List of matched evidence dictionaries with component weights and descriptions.
        missing_evidence: List of missing evidence dictionaries with component weights and descriptions.
        behaviour_evidence: List of matched Behaviour Graph pattern names.
        action_evidence: List of matched Action Recognition labels.
        motion_evidence: Dictionary of matched motion statistics.
        spatial_evidence: Dictionary of matched spatial relationship statistics.
        temporal_evidence: Dictionary of matched temporal sequence metrics.
        evidence_timeline: Chronological list of synchronized multi-modal evidence events.
        explanation_text: Human-readable forensic explanation string with evidence checkmarks.
        recommendation: Recommended action for human forensic investigators.
        metadata: Arbitrary metadata.
    """

    signature_id: str = field(default_factory=lambda: f"SIG-{uuid.uuid4().hex[:8].upper()}")
    interaction_id: str = ""
    fusion_id: str = ""
    matched_signature_name: str = "StandardMotorcycleSnatch"

    signature_score: float = 0.0
    decision: str = "No Match"
    confidence: float = 0.0

    matched_evidence: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = field(default_factory=list)

    behaviour_evidence: list[str] = field(default_factory=list)
    action_evidence: list[str] = field(default_factory=list)

    motion_evidence: dict[str, Any] = field(default_factory=dict)
    spatial_evidence: dict[str, Any] = field(default_factory=dict)
    temporal_evidence: dict[str, Any] = field(default_factory=dict)

    evidence_timeline: list[dict[str, Any]] = field(default_factory=list)
    explanation_text: str = ""
    recommendation: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

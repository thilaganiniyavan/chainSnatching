"""Unit tests for the SignatureMatcher module.

Tests cover:
- Multi-modal evidence weighted scoring
- Decision boundary assignment (High Confidence, Strong, Partial, Weak, No Match)
- Matched vs missing evidence compilation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.fused_interaction import FusedInteraction
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.snatch.signature_config import StandardMotorcycleSnatchSignature
from src.snatch.signature_matcher import SignatureMatcher


# ======================================================================
# Helpers
# ======================================================================

def _make_fusion(patterns=None, actions=None, speed: float = 0.10, dist: float = 50.0) -> FusedInteraction:
    pats = patterns if patterns is not None else ["APPROACH_PATTERN", "INTERACTION_PATTERN", "ESCAPE_PATTERN"]
    acts = actions if actions is not None else [{"action_label": "Reaching", "action_confidence": 0.90}]

    return FusedInteraction(
        fusion_id="FUSED-INT-001",
        interaction_id="INT-001",
        person_track_id=1,
        vehicle_track_id=2,
        behaviour_patterns=pats,
        action_timeline=acts,
        motion_evidence={"average_speed_px": speed},
        spatial_evidence={"min_distance_px": dist},
        fusion_confidence=0.90,
    )


# ======================================================================
# SignatureMatcher Tests
# ======================================================================

class TestSignatureMatcher:

    def test_evaluate_high_confidence_match(self):
        matcher = SignatureMatcher(template=StandardMotorcycleSnatchSignature())
        fusion = _make_fusion() # matches approach, interaction, reaching action, speed, escape, proximity

        res = matcher.evaluate(fusion)
        assert isinstance(res, SnatchSignatureResult)
        assert res.signature_score == 1.0
        assert res.decision == "High Confidence Match"
        assert len(res.matched_evidence) == 6
        assert len(res.missing_evidence) == 0

    def test_evaluate_partial_match(self):
        matcher = SignatureMatcher(template=StandardMotorcycleSnatchSignature())
        # Only approach and interaction pattern present, target action and escape missing
        fusion = _make_fusion(
            patterns=["APPROACH_PATTERN", "INTERACTION_PATTERN"],
            actions=[{"action_label": "Walking", "action_confidence": 0.80}],
            speed=0.01,
            dist=200.0,
        )

        res = matcher.evaluate(fusion)
        assert res.decision in ("Weak Match", "Partial Match")
        assert len(res.missing_evidence) > 0

    def test_evaluate_no_match(self):
        matcher = SignatureMatcher(template=StandardMotorcycleSnatchSignature())
        fusion = _make_fusion(
            patterns=[],
            actions=[],
            speed=0.0,
            dist=500.0,
        )

        res = matcher.evaluate(fusion)
        assert res.signature_score == 0.0
        assert res.decision == "No Match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

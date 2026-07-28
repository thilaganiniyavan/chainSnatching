"""Unit tests for the SnatchSignatureEngine and SignatureExplainer modules.

Tests cover:
- Single interaction and batch signature evaluation
- SnatchSignatureResult field population
- Searching and filtering flagged results by threshold
- Forensic evidence explanation and recommendation formatting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.fused_interaction import FusedInteraction
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.snatch.snatch_signature_engine import SnatchSignatureEngine
from src.snatch.signature_explainer import SignatureExplainer


# ======================================================================
# Helpers
# ======================================================================

def _make_fusion(iid: str = "INT-001") -> FusedInteraction:
    return FusedInteraction(
        fusion_id=f"FUSED-{iid}",
        interaction_id=iid,
        person_track_id=1,
        vehicle_track_id=2,
        behaviour_patterns=["APPROACH_PATTERN", "INTERACTION_PATTERN", "ESCAPE_PATTERN"],
        action_timeline=[{"action_label": "Reaching", "action_confidence": 0.90}],
        motion_evidence={"average_speed_px": 0.15},
        spatial_evidence={"min_distance_px": 45.0},
        fusion_confidence=0.88,
    )


# ======================================================================
# SnatchSignatureEngine Tests
# ======================================================================

class TestSnatchSignatureEngine:

    def test_evaluate_interaction(self):
        engine = SnatchSignatureEngine()
        fusion = _make_fusion("INT-001")

        res = engine.evaluate_interaction(fusion)
        assert isinstance(res, SnatchSignatureResult)
        assert res.interaction_id == "INT-001"
        assert res.signature_score == 1.0
        assert res.decision == "High Confidence Match"
        assert "✓" in res.explanation_text
        assert "FLAGGED FOR HIGH-PRIORITY FORENSIC REVIEW" in res.recommendation

    def test_evaluate_batch_and_get_flagged_results(self):
        engine = SnatchSignatureEngine()
        f1 = _make_fusion("INT-001")
        f2 = FusedInteraction(
            fusion_id="FUSED-INT-002",
            interaction_id="INT-002",
            behaviour_patterns=[],
        )

        results = engine.evaluate_batch([f1, f2])
        assert len(results) == 2

        flagged = engine.get_flagged_results(min_score=0.70)
        assert len(flagged) == 1
        assert flagged[0].interaction_id == "INT-001"

    def test_signature_explainer_formatting(self):
        explainer = SignatureExplainer()
        res = SnatchSignatureResult(
            signature_id="SIG-001",
            interaction_id="INT-001",
            matched_signature_name="StandardMotorcycleSnatch",
            signature_score=0.85,
            decision="High Confidence Match",
            matched_evidence=[{"description": "Approach pattern observed"}],
            missing_evidence=[{"description": "Confirmed grabbing action missing"}],
        )

        exp_text, rec_text = explainer.format_explanation(res)
        assert "StandardMotorcycleSnatch" in exp_text
        assert "✓ Approach pattern observed" in exp_text
        assert "✗ Confirmed grabbing action missing" in exp_text
        assert "FLAGGED FOR HIGH-PRIORITY FORENSIC REVIEW" in rec_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

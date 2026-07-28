"""Unit tests for the ForensicQueryEngine module.

Tests cover:
- add_event, update_event, delete_event, get_event APIs
- Expressive search_events query text matching
- Multi-criteria filter_events (decision, min_confidence, min_score, track_id, pattern, action)
- Construction of ForensicEvent records from SnatchSignatureResult
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.forensic_event import ForensicEvent
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.forensic.forensic_query_engine import ForensicQueryEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_sig_result() -> SnatchSignatureResult:
    return SnatchSignatureResult(
        signature_id="SIG-001",
        interaction_id="INT-001",
        fusion_id="FUSED-INT-001",
        matched_signature_name="StandardMotorcycleSnatch",
        signature_score=0.92,
        decision="High Confidence Match",
        confidence=0.90,
        matched_evidence=[{"description": "Approach pattern"}],
        missing_evidence=[],
        behaviour_evidence=["APPROACH_PATTERN", "INTERACTION_PATTERN", "ESCAPE_PATTERN"],
        action_evidence=["Reaching", "Grabbing"],
        metadata={"person_track_id": 5, "vehicle_track_id": 2},
    )


# ======================================================================
# ForensicQueryEngine Tests
# ======================================================================

class TestForensicQueryEngine:

    def test_create_event_from_signature(self):
        engine = ForensicQueryEngine()
        sig = _make_sig_result()

        event = engine.create_event_from_signature(sig, video_id="test.mp4")
        assert isinstance(event, ForensicEvent)
        assert event.event_id == "EVT-SIG-001"
        assert event.decision == "High Confidence Match"
        assert event.person_track_id == 5
        assert "APPROACH_PATTERN" in event.behaviour_patterns
        assert "Reaching" in event.detected_actions

    def test_search_events(self):
        engine = ForensicQueryEngine()
        sig1 = _make_sig_result()
        engine.create_event_from_signature(sig1)

        # Query token matching
        results = engine.search_events("Grabbing")
        assert len(results) == 1
        assert results[0].event_id == "EVT-SIG-001"

        no_results = engine.search_events("NonExistentAction")
        assert len(no_results) == 0

    def test_filter_events(self):
        engine = ForensicQueryEngine()
        sig1 = _make_sig_result()
        engine.create_event_from_signature(sig1)

        # Filter by decision and score threshold
        filtered = engine.filter_events(
            decision="High Confidence Match",
            min_score=0.85,
            track_id=5,
            action="Reaching",
        )
        assert len(filtered) == 1

        # Filter out by higher threshold
        empty_filtered = engine.filter_events(min_score=0.99)
        assert len(empty_filtered) == 0

    def test_update_and_delete_event(self):
        engine = ForensicQueryEngine()
        sig = _make_sig_result()
        event = engine.create_event_from_signature(sig)

        # Update
        updated = engine.update_event(event.event_id, {"investigator_notes": "Reviewed and verified."})
        assert updated is not None
        assert updated.investigator_notes == "Reviewed and verified."

        # Delete
        assert engine.delete_event(event.event_id) is True
        assert engine.get_event(event.event_id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

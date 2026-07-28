"""Unit tests for the ForensicIndexEngine module.

Tests cover:
- Indexing ForensicEvent records across inverted indices
- Fast O(1) multi-attribute lookups (decision, track_id, pattern, action, signature_name, tag)
- Intersecting queries across multiple criteria
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.forensic_event import ForensicEvent
from src.forensic.forensic_index_engine import ForensicIndexEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_event(
    event_id: str = "EVT-001",
    decision: str = "High Confidence Match",
    track_id: int = 1,
    pattern: str = "APPROACH_PATTERN",
    action: str = "Reaching",
) -> ForensicEvent:
    return ForensicEvent(
        event_id=event_id,
        video_id="test_video.mp4",
        interaction_id="INT-001",
        person_track_id=track_id,
        vehicle_track_id=2,
        decision=decision,
        signature_score=0.95,
        confidence=0.90,
        matched_signature_name="StandardMotorcycleSnatch",
        behaviour_patterns=[pattern, "INTERACTION_PATTERN"],
        detected_actions=[action],
        tags=["Snatch", pattern, action],
    )


# ======================================================================
# ForensicIndexEngine Tests
# ======================================================================

class TestForensicIndexEngine:

    def test_index_and_get_event(self):
        engine = ForensicIndexEngine()
        event = _make_event("EVT-001")

        engine.index_event(event)
        retrieved = engine.get_event("EVT-001")
        assert retrieved is not None
        assert retrieved.event_id == "EVT-001"
        assert retrieved.decision == "High Confidence Match"

    def test_lookup_by_decision(self):
        engine = ForensicIndexEngine()
        engine.index_event(_make_event("EVT-001", decision="High Confidence Match"))
        engine.index_event(_make_event("EVT-002", decision="No Match"))

        matches = engine.lookup(decision="High Confidence Match")
        assert matches == {"EVT-001"}

    def test_lookup_by_track_id(self):
        engine = ForensicIndexEngine()
        engine.index_event(_make_event("EVT-001", track_id=5))
        engine.index_event(_make_event("EVT-002", track_id=10))

        matches = engine.lookup(track_id=5)
        assert "EVT-001" in matches
        assert "EVT-002" not in matches

    def test_lookup_by_pattern_and_action(self):
        engine = ForensicIndexEngine()
        engine.index_event(_make_event("EVT-001", pattern="ESCAPE_PATTERN", action="Grabbing"))
        engine.index_event(_make_event("EVT-002", pattern="APPROACH_PATTERN", action="Walking"))

        matches = engine.lookup(pattern="ESCAPE_PATTERN", action="Grabbing")
        assert matches == {"EVT-001"}

    def test_remove_event(self):
        engine = ForensicIndexEngine()
        event = _make_event("EVT-001")
        engine.index_event(event)

        assert engine.remove_event("EVT-001") is True
        assert engine.get_event("EVT-001") is None
        assert len(engine.lookup(decision="High Confidence Match")) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

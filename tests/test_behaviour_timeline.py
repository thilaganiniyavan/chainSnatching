"""Unit tests for the BehaviourTimeline.

Tests cover:
- Timeline recording from interactions and behaviour primitives
- Lifecycle event generation
- Human-readable formatting
- Dictionary serialization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.behavior.behaviour_timeline import BehaviourTimeline, TimelineEvent


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction(
    interaction_id: str = "INT-0001",
    state: InteractionState = InteractionState.NEW,
    current_distance: float = 100.0,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        current_frame=5,
        duration=5,
        current_distance=current_distance,
        state=state,
    )


def _make_behaviour(
    interaction_id: str = "INT-0001",
    primitive_type: str = "APPROACHING",
    confidence: float = 0.8,
) -> BehaviourPrimitive:
    return BehaviourPrimitive(
        primitive_type=primitive_type,
        interaction_id=interaction_id,
        start_frame=5,
        end_frame=5,
        confidence=confidence,
        measurements={"relative_velocity": -5.0},
    )


# ======================================================================
# Tests — Recording
# ======================================================================

class TestRecording:
    def test_records_lifecycle_event(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.NEW)

        timeline.record(interaction, [], frame_number=5)

        events = timeline.get_timeline("INT-0001")
        assert len(events) == 1
        assert events[0].event_type == "INTERACTION_STARTED"

    def test_records_behaviour_primitive(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.ACTIVE)
        behaviour = _make_behaviour()

        # First record NEW lifecycle
        timeline.record(
            _make_interaction(state=InteractionState.NEW), [], frame_number=1
        )
        # Then record ACTIVE + behaviour
        timeline.record(interaction, [behaviour], frame_number=5)

        events = timeline.get_timeline("INT-0001")
        event_types = [e.event_type for e in events]
        assert "INTERACTION_STARTED" in event_types
        assert "INTERACTION_ACTIVE" in event_types
        assert "APPROACHING" in event_types

    def test_lifecycle_recorded_once(self):
        """Each lifecycle state should only generate one timeline event."""
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.ACTIVE)

        timeline.record(interaction, [], frame_number=5)
        timeline.record(interaction, [], frame_number=6)
        timeline.record(interaction, [], frame_number=7)

        events = timeline.get_timeline("INT-0001")
        lifecycle_events = [
            e for e in events if e.event_type.startswith("INTERACTION_")
        ]
        assert len(lifecycle_events) == 1

    def test_multiple_behaviours_same_frame(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.ACTIVE)
        b1 = _make_behaviour(primitive_type="APPROACHING")
        b2 = _make_behaviour(primitive_type="CLOSE_INTERACTION")

        timeline.record(interaction, [b1, b2], frame_number=10)

        events = timeline.get_timeline("INT-0001")
        behaviour_types = [
            e.event_type for e in events
            if not e.event_type.startswith("INTERACTION_")
        ]
        assert "APPROACHING" in behaviour_types
        assert "CLOSE_INTERACTION" in behaviour_types

    def test_behaviours_filtered_by_interaction_id(self):
        """Behaviours for other interactions should be ignored."""
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(interaction_id="INT-0001")
        other_behaviour = _make_behaviour(
            interaction_id="INT-9999", primitive_type="RAPID_SEPARATION"
        )

        timeline.record(interaction, [other_behaviour], frame_number=5)

        events = timeline.get_timeline("INT-0001")
        types = [e.event_type for e in events]
        assert "RAPID_SEPARATION" not in types


# ======================================================================
# Tests — Timestamp Computation
# ======================================================================

class TestTimestamps:
    def test_timestamp_from_fps(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.NEW)
        timeline.record(interaction, [], frame_number=90)

        events = timeline.get_timeline("INT-0001")
        assert len(events) == 1
        # 90 / 30 = 3.0 seconds
        assert events[0].timestamp == 3.0

    def test_default_fps_fallback(self):
        """FPS <= 0 should fall back to 30.0."""
        timeline = BehaviourTimeline(fps=-5.0)
        assert timeline.fps == 30.0


# ======================================================================
# Tests — Formatting
# ======================================================================

class TestFormatting:
    def test_format_timeline_contains_events(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.NEW)
        timeline.record(interaction, [], frame_number=90)

        text = timeline.format_timeline("INT-0001")
        assert "INT-0001" in text
        assert "INTERACTION_STARTED" in text

    def test_format_unknown_interaction(self):
        timeline = BehaviourTimeline()
        text = timeline.format_timeline("DOES-NOT-EXIST")
        assert "No timeline recorded" in text

    def test_format_all_timelines(self):
        timeline = BehaviourTimeline(fps=30.0)
        for iid in ["INT-0001", "INT-0002"]:
            interaction = _make_interaction(
                interaction_id=iid, state=InteractionState.NEW
            )
            timeline.record(interaction, [], frame_number=10)

        text = timeline.format_all_timelines()
        assert "INT-0001" in text
        assert "INT-0002" in text


# ======================================================================
# Tests — Serialization
# ======================================================================

class TestSerialization:
    def test_to_dict(self):
        timeline = BehaviourTimeline(fps=30.0)
        interaction = _make_interaction(state=InteractionState.NEW)
        behaviour = _make_behaviour()
        timeline.record(interaction, [behaviour], frame_number=5)

        result = timeline.to_dict("INT-0001")
        assert isinstance(result, list)
        assert len(result) >= 1
        first = result[0]
        assert "frame_number" in first
        assert "timestamp" in first
        assert "event_type" in first

    def test_to_dict_empty(self):
        timeline = BehaviourTimeline()
        result = timeline.to_dict("NOPE")
        assert result == []


# ======================================================================
# Tests — get_all_timelines
# ======================================================================

class TestGetAllTimelines:
    def test_returns_all(self):
        timeline = BehaviourTimeline(fps=30.0)
        for iid in ["INT-A", "INT-B", "INT-C"]:
            interaction = _make_interaction(
                interaction_id=iid, state=InteractionState.NEW
            )
            timeline.record(interaction, [], frame_number=1)

        all_tl = timeline.get_all_timelines()
        assert set(all_tl.keys()) == {"INT-A", "INT-B", "INT-C"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

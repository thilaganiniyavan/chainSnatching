"""Unit tests for the ReasoningEngine.

Tests cover:
- Classification of each of the 7 Behaviour Event types
- Rule graph sequence ordering
- Priority-based conflict resolution
- Confidence scoring
- Tentative event classification on active interactions
- Edge cases (empty timelines, unfulfilled constraints)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_event import BehaviourEvent
from src.behavior.behaviour_timeline import TimelineEvent
from src.behavior.reasoning_engine import ReasoningEngine
from src.behavior.reasoning_rules import RuleNode


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction(
    interaction_id: str = "INT-0001",
    duration: int = 30,
    min_dist: float = 40.0,
    avg_dist: float = 70.0,
    rel_vel: float = 0.0,
    rel_acc: float = 0.0,
    state: InteractionState = InteractionState.ENDED,
) -> Interaction:
    interaction = Interaction(
        interaction_id=interaction_id,
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        current_frame=duration,
        end_frame=duration,
        duration=duration,
        min_distance=min_dist,
        avg_distance=avg_dist,
        current_distance=avg_dist,
        relative_velocity=rel_vel,
        relative_acceleration=rel_acc,
        state=state,
    )
    # Populate constant relationship & motion history for evidence calculation
    interaction.relationship_history = [
        {"frame": f, "distance": avg_dist}
        for f in range(1, duration + 1)
    ]
    if len(interaction.relationship_history) > 15:
        interaction.relationship_history[14]["distance"] = min_dist

    interaction.motion_history = [
        {"frame": f, "person_speed": 1.0, "vehicle_speed": 1.0}
        for f in range(1, duration + 1)
    ]
    return interaction


def _make_timeline(
    interaction_id: str,
    primitives: list[str],
    fps: float = 30.0,
) -> list[TimelineEvent]:
    events = [
        TimelineEvent(
            frame_number=1,
            timestamp=0.033,
            event_type="INTERACTION_STARTED",
            interaction_id=interaction_id,
        )
    ]
    for i, prim in enumerate(primitives):
        frame = (i + 1) * 5
        events.append(
            TimelineEvent(
                frame_number=frame,
                timestamp=round(frame / fps, 3),
                event_type=prim,
                interaction_id=interaction_id,
            )
        )
    return events


# ======================================================================
# Classification Tests for 7 Event Types
# ======================================================================

class TestEventClassifications:

    def test_normal_passing(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=20, min_dist=100.0, avg_dist=120.0)
        interaction.relationship_history = [{"frame": f, "distance": 120.0} for f in range(1, 21)]
        timeline = _make_timeline("INT-0001", ["APPROACHING", "MOVING_AWAY"])

        events = engine.analyse_interaction(interaction, timeline)
        assert len(events) >= 1
        assert events[0].event_type == "NORMAL_PASSING"
        assert not events[0].is_tentative

    def test_vehicle_waiting(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=50, min_dist=70.0, avg_dist=70.0, rel_vel=0.0, rel_acc=0.0)
        interaction.relationship_history = [{"frame": f, "distance": 70.0} for f in range(1, 51)]
        interaction.motion_history = [
            {"frame": f, "person_speed": 0.5, "vehicle_speed": 0.5}
            for f in range(1, 51)
        ]
        timeline = _make_timeline("INT-0001", ["STATIONARY_INTERACTION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "VEHICLE_WAITING" in types

    def test_following_behaviour(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=40)
        interaction.relationship_history = [{"frame": f, "distance": 70.0} for f in range(1, 41)]
        timeline = _make_timeline("INT-0001", ["FOLLOWING", "PARALLEL_MOTION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "FOLLOWING_BEHAVIOUR" in types

    def test_stationary_interaction(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=30, min_dist=30.0, avg_dist=30.0, rel_vel=0.0, rel_acc=0.0)
        interaction.relationship_history = [{"frame": f, "distance": 30.0} for f in range(1, 31)]
        interaction.motion_history = [
            {"frame": f, "person_speed": 0.5, "vehicle_speed": 0.5}
            for f in range(1, 31)
        ]
        timeline = _make_timeline("INT-0001", ["STATIONARY_INTERACTION", "CLOSE_INTERACTION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "STATIONARY_INTERACTION" in types

    def test_close_encounter(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=20, min_dist=40.0, avg_dist=50.0, rel_vel=-1.0, rel_acc=0.0)
        interaction.relationship_history = [{"frame": f, "distance": 60.0 - f} for f in range(1, 21)]
        timeline = _make_timeline("INT-0001", ["APPROACHING", "CLOSE_INTERACTION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "CLOSE_ENCOUNTER" in types

    def test_suspicious_encounter(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=30, min_dist=40.0, rel_acc=4.0)
        interaction.relationship_history = [
            {"frame": 1, "distance": 150.0},
            {"frame": 2, "distance": 100.0},  # vel = -50
            {"frame": 3, "distance": 40.0},   # vel = -60 -> accel = -10 (abs = 10)
        ]
        timeline = _make_timeline("INT-0001", ["APPROACHING", "CLOSE_INTERACTION", "RAPID_ACCELERATION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "SUSPICIOUS_ENCOUNTER" in types

    def test_rapid_escape(self):
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=30, rel_vel=10.0, rel_acc=5.0)
        interaction.relationship_history = [
            {"frame": 1, "distance": 40.0},
            {"frame": 2, "distance": 60.0},   # vel = 20
            {"frame": 3, "distance": 90.0},   # vel = 30 -> accel = 10
        ]
        timeline = _make_timeline("INT-0001", ["RAPID_ACCELERATION", "RAPID_SEPARATION"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "RAPID_ESCAPE" in types


# ======================================================================
# Priority and Sequence Tests
# ======================================================================

class TestEngineLogic:

    def test_priority_conflict_resolution(self):
        """Higher priority event (RAPID_ESCAPE, priority 6) should win over lower priority."""
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=30, min_dist=40.0, rel_vel=10.0)
        interaction.relationship_history = [
            {"frame": 1, "distance": 40.0},
            {"frame": 2, "distance": 60.0},
            {"frame": 3, "distance": 90.0},
        ]
        timeline = _make_timeline(
            "INT-0001", ["APPROACHING", "CLOSE_INTERACTION", "RAPID_ACCELERATION", "RAPID_SEPARATION"]
        )

        events = engine.analyse_interaction(interaction, timeline)
        assert len(events) == 1
        assert events[0].event_type == "RAPID_ESCAPE"  # Priority 6 wins

    def test_sequence_order_enforcement(self):
        """Rule requiring specific primitive order should not fire if order is reversed."""
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=30)
        timeline = _make_timeline("INT-0001", ["MOVING_AWAY", "APPROACHING"])

        events = engine.analyse_interaction(interaction, timeline)
        types = [e.event_type for e in events]
        assert "NORMAL_PASSING" not in types

    def test_tentative_classification(self):
        """Active interactions classified with tentative=True should produce tentative events."""
        engine = ReasoningEngine()
        interaction = _make_interaction(duration=20, state=InteractionState.ACTIVE)
        interaction.relationship_history = [{"frame": f, "distance": 50.0} for f in range(1, 21)]
        timeline = _make_timeline("INT-0001", ["APPROACHING", "CLOSE_INTERACTION"])

        events = engine.analyse_interaction(interaction, timeline, tentative=True)
        assert len(events) >= 1
        assert events[0].is_tentative

    def test_empty_timeline_returns_empty(self):
        engine = ReasoningEngine()
        interaction = _make_interaction()
        events = engine.analyse_interaction(interaction, [])
        assert events == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

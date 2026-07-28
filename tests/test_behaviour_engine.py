"""Unit tests for the BehaviourEngine.

Tests cover each of the 11 individual primitive detectors with synthetic
Interaction objects, plus edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction, InteractionState
from src.behavior.behaviour_engine import BehaviourEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction(**overrides) -> Interaction:
    """Create a minimal Interaction with sensible defaults, allowing overrides."""
    defaults = dict(
        interaction_id="INT-TEST",
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        current_frame=10,
        duration=10,
        min_distance=30.0,
        max_distance=120.0,
        avg_distance=75.0,
        current_distance=80.0,
        relative_velocity=0.0,
        relative_acceleration=0.0,
        heading_difference=0.0,
        trajectory_similarity=0.0,
        interaction_confidence=0.5,
        state=InteractionState.ACTIVE,
        motion_history=[],
        relationship_history=[],
    )
    defaults.update(overrides)
    return Interaction(**defaults)


# ======================================================================
# Tests — APPROACHING
# ======================================================================

class TestApproaching:
    def test_detects_approaching(self):
        engine = BehaviourEngine(approach_velocity_threshold=-2.0, close_distance_threshold=80.0)
        interaction = _make_interaction(
            relative_velocity=-5.0,
            current_distance=120.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "APPROACHING" in types

    def test_not_approaching_when_close(self):
        """Should NOT fire when already within close distance."""
        engine = BehaviourEngine(close_distance_threshold=80.0)
        interaction = _make_interaction(
            relative_velocity=-5.0,
            current_distance=50.0,  # Already close
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "APPROACHING" not in types

    def test_not_approaching_when_separating(self):
        engine = BehaviourEngine()
        interaction = _make_interaction(relative_velocity=3.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "APPROACHING" not in types


# ======================================================================
# Tests — MOVING_AWAY
# ======================================================================

class TestMovingAway:
    def test_detects_moving_away(self):
        engine = BehaviourEngine(close_distance_threshold=80.0)
        interaction = _make_interaction(
            relative_velocity=3.0,
            current_distance=120.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "MOVING_AWAY" in types

    def test_not_moving_away_when_close(self):
        engine = BehaviourEngine(close_distance_threshold=80.0)
        interaction = _make_interaction(
            relative_velocity=3.0,
            current_distance=50.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "MOVING_AWAY" not in types


# ======================================================================
# Tests — FOLLOWING
# ======================================================================

class TestFollowing:
    def test_detects_following(self):
        engine = BehaviourEngine(
            following_similarity_threshold=0.7,
            following_min_frames=5,
            parallel_heading_threshold=15.0,
        )
        interaction = _make_interaction(
            heading_difference=10.0,
            trajectory_similarity=0.85,
            duration=10,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "FOLLOWING" in types

    def test_not_following_short_duration(self):
        engine = BehaviourEngine(following_min_frames=5)
        interaction = _make_interaction(
            heading_difference=10.0,
            trajectory_similarity=0.9,
            duration=3,  # Too short
        )
        results = engine.analyse([interaction], frame_number=3)
        types = [r.primitive_type for r in results]
        assert "FOLLOWING" not in types


# ======================================================================
# Tests — CLOSE_INTERACTION
# ======================================================================

class TestCloseInteraction:
    def test_detects_close_interaction(self):
        engine = BehaviourEngine(close_distance_threshold=80.0)
        interaction = _make_interaction(current_distance=30.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "CLOSE_INTERACTION" in types

    def test_not_close_when_far(self):
        engine = BehaviourEngine(close_distance_threshold=80.0)
        interaction = _make_interaction(current_distance=100.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "CLOSE_INTERACTION" not in types


# ======================================================================
# Tests — RAPID_ACCELERATION / DECELERATION
# ======================================================================

class TestAcceleration:
    def test_detects_rapid_acceleration(self):
        engine = BehaviourEngine(acceleration_threshold=3.0)
        interaction = _make_interaction(relative_acceleration=5.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "RAPID_ACCELERATION" in types

    def test_detects_rapid_deceleration(self):
        engine = BehaviourEngine(acceleration_threshold=3.0)
        interaction = _make_interaction(relative_acceleration=-5.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "RAPID_DECELERATION" in types

    def test_no_acceleration_below_threshold(self):
        engine = BehaviourEngine(acceleration_threshold=3.0)
        interaction = _make_interaction(relative_acceleration=1.0)
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "RAPID_ACCELERATION" not in types
        assert "RAPID_DECELERATION" not in types


# ======================================================================
# Tests — RAPID_SEPARATION
# ======================================================================

class TestRapidSeparation:
    def test_detects_rapid_separation(self):
        engine = BehaviourEngine(separation_velocity_threshold=5.0)
        interaction = _make_interaction(
            relative_velocity=8.0,
            current_distance=100.0,
            avg_distance=60.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "RAPID_SEPARATION" in types

    def test_no_separation_below_threshold(self):
        engine = BehaviourEngine(separation_velocity_threshold=5.0)
        interaction = _make_interaction(
            relative_velocity=2.0,
            current_distance=100.0,
            avg_distance=60.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "RAPID_SEPARATION" not in types


# ======================================================================
# Tests — STATIONARY_INTERACTION
# ======================================================================

class TestStationaryInteraction:
    def test_detects_stationary(self):
        engine = BehaviourEngine(
            stationary_speed_threshold=2.0,
            close_distance_threshold=80.0,
        )
        interaction = _make_interaction(
            current_distance=40.0,
            motion_history=[
                {"person_speed": 0.5, "vehicle_speed": 0.3}
            ],
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "STATIONARY_INTERACTION" in types

    def test_not_stationary_when_moving(self):
        engine = BehaviourEngine(stationary_speed_threshold=2.0)
        interaction = _make_interaction(
            current_distance=40.0,
            motion_history=[
                {"person_speed": 10.0, "vehicle_speed": 8.0}
            ],
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "STATIONARY_INTERACTION" not in types


# ======================================================================
# Tests — SUDDEN_DIRECTION_CHANGE
# ======================================================================

class TestSuddenDirectionChange:
    def test_detects_direction_change(self):
        engine = BehaviourEngine(heading_change_threshold=45.0)
        interaction = _make_interaction(
            motion_history=[
                {"person_direction": 90.0, "vehicle_direction": 100.0},
                {"person_direction": 90.0, "vehicle_direction": 200.0},
            ],
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "SUDDEN_DIRECTION_CHANGE" in types

    def test_no_change_below_threshold(self):
        engine = BehaviourEngine(heading_change_threshold=45.0)
        interaction = _make_interaction(
            motion_history=[
                {"person_direction": 90.0, "vehicle_direction": 100.0},
                {"person_direction": 90.0, "vehicle_direction": 105.0},
            ],
        )
        results = engine.analyse([interaction], frame_number=10)
        types = [r.primitive_type for r in results]
        assert "SUDDEN_DIRECTION_CHANGE" not in types


# ======================================================================
# Tests — INTERACTION_DURATION
# ======================================================================

class TestInteractionDuration:
    def test_emits_at_bucket_boundary(self):
        engine = BehaviourEngine(duration_bucket_size=30)
        interaction = _make_interaction(duration=30, start_frame=1)
        results = engine.analyse([interaction], frame_number=30)
        types = [r.primitive_type for r in results]
        assert "INTERACTION_DURATION" in types

    def test_no_emission_between_buckets(self):
        engine = BehaviourEngine(duration_bucket_size=30)
        interaction = _make_interaction(duration=25, start_frame=1)
        results = engine.analyse([interaction], frame_number=25)
        types = [r.primitive_type for r in results]
        assert "INTERACTION_DURATION" not in types


# ======================================================================
# Tests — Edge Cases
# ======================================================================

class TestEdgeCases:
    def test_ended_interactions_ignored(self):
        engine = BehaviourEngine()
        interaction = _make_interaction(
            state=InteractionState.ENDED,
            relative_velocity=-10.0,
            current_distance=120.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        assert len(results) == 0

    def test_empty_interactions(self):
        engine = BehaviourEngine()
        results = engine.analyse([], frame_number=10)
        assert results == []

    def test_zero_velocity(self):
        engine = BehaviourEngine()
        interaction = _make_interaction(relative_velocity=0.0)
        results = engine.analyse([interaction], frame_number=10)
        # Should not crash
        assert isinstance(results, list)

    def test_confidence_bounds(self):
        engine = BehaviourEngine()
        interaction = _make_interaction(
            relative_velocity=-100.0,
            current_distance=200.0,
        )
        results = engine.analyse([interaction], frame_number=10)
        for bp in results:
            assert 0.0 <= bp.confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

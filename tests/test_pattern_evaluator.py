"""Unit tests for the PatternEvaluator module.

Tests cover detection of each of the 11 Behaviour Patterns:
- APPROACH_PATTERN
- FOLLOW_PATTERN
- CO_TRAVEL_PATTERN
- PROXIMITY_PATTERN
- INTERACTION_PATTERN
- STOP_PATTERN
- LINGERING_PATTERN
- SEPARATION_PATTERN
- ESCAPE_PATTERN
- DIVERGENCE_PATTERN
- WAITING_PATTERN
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.behavior.pattern_evaluator import PatternEvaluator
from src.behavior.pattern_rules import PatternConfig


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction(
    interaction_id: str = "INT-0001",
    duration: int = 20,
    min_dist: float = 40.0,
    avg_dist: float = 70.0,
    curr_dist: float = 70.0,
    rel_vel: float = 0.0,
    rel_acc: float = 0.0,
    heading_diff: float = 0.0,
    traj_sim: float = 0.0,
    state: InteractionState = InteractionState.ACTIVE,
) -> Interaction:
    interaction = Interaction(
        interaction_id=interaction_id,
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        current_frame=duration,
        end_frame=None,
        duration=duration,
        min_distance=min_dist,
        avg_distance=avg_dist,
        current_distance=curr_dist,
        relative_velocity=rel_vel,
        relative_acceleration=rel_acc,
        heading_difference=heading_diff,
        trajectory_similarity=traj_sim,
        state=state,
    )
    interaction.motion_history = [
        {"person_speed": 1.0, "vehicle_speed": 1.0} for _ in range(duration)
    ]
    return interaction


def _make_primitive(ptype: str, iid: str = "INT-0001") -> BehaviourPrimitive:
    return BehaviourPrimitive(
        primitive_type=ptype,
        interaction_id=iid,
        start_frame=1,
        end_frame=10,
        confidence=0.8,
    )


# ======================================================================
# Pattern Evaluator Tests
# ======================================================================

class TestPatternEvaluator:

    def test_approach_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(rel_vel=-3.0)
        prims = [_make_primitive("APPROACHING")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "APPROACH_PATTERN" in types

    def test_follow_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(duration=15, traj_sim=0.8, heading_diff=10.0)
        prims = [_make_primitive("FOLLOWING")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=15)
        types = [n.pattern_type for n in nodes]
        assert "FOLLOW_PATTERN" in types

    def test_co_travel_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(heading_diff=10.0)
        prims = [_make_primitive("PARALLEL_MOTION")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "CO_TRAVEL_PATTERN" in types

    def test_proximity_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(curr_dist=40.0)
        prims = [_make_primitive("CLOSE_INTERACTION")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "PROXIMITY_PATTERN" in types

    def test_interaction_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(min_dist=30.0, duration=10)

        nodes = evaluator.evaluate(interaction, [], [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "INTERACTION_PATTERN" in types

    def test_stop_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(curr_dist=50.0)
        interaction.motion_history = [{"person_speed": 0.5, "vehicle_speed": 0.5}]
        prims = [_make_primitive("STATIONARY_INTERACTION")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "STOP_PATTERN" in types

    def test_lingering_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(
            duration=20, avg_dist=50.0, state=InteractionState.LINGERING
        )

        nodes = evaluator.evaluate(interaction, [], [], frame_number=20)
        types = [n.pattern_type for n in nodes]
        assert "LINGERING_PATTERN" in types

    def test_separation_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(rel_vel=4.0)
        prims = [_make_primitive("MOVING_AWAY")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "SEPARATION_PATTERN" in types

    def test_escape_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(rel_vel=5.0, rel_acc=3.0)
        prims = [
            _make_primitive("RAPID_ACCELERATION"),
            _make_primitive("RAPID_SEPARATION"),
        ]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "ESCAPE_PATTERN" in types

    def test_divergence_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(heading_diff=60.0)
        prims = [_make_primitive("SUDDEN_DIRECTION_CHANGE")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=10)
        types = [n.pattern_type for n in nodes]
        assert "DIVERGENCE_PATTERN" in types

    def test_waiting_pattern(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(duration=20)
        interaction.motion_history = [{"person_speed": 1.0, "vehicle_speed": 0.5}]
        prims = [_make_primitive("STATIONARY_INTERACTION")]

        nodes = evaluator.evaluate(interaction, prims, [], frame_number=20)
        types = [n.pattern_type for n in nodes]
        assert "WAITING_PATTERN" in types

    def test_archived_interaction_returns_empty(self):
        evaluator = PatternEvaluator()
        interaction = _make_interaction(state=InteractionState.ARCHIVED)

        nodes = evaluator.evaluate(interaction, [], [], frame_number=20)
        assert nodes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

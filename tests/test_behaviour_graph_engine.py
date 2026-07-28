"""Unit tests for the BehaviourGraphEngine module.

Tests cover:
- Graph creation and initialization
- Per-frame graph updates and PatternNode creation
- TransitionEdge generation between consecutive pattern nodes
- Graph finalization
- Query APIs (get_graph, get_active_graphs, get_completed_graphs, find_patterns)
- Transition statistics computation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.behavior.behaviour_graph_engine import BehaviourGraphEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction(
    interaction_id: str = "INT-0001",
    duration: int = 10,
    curr_dist: float = 70.0,
    rel_vel: float = 0.0,
    rel_acc: float = 0.0,
    heading_diff: float = 0.0,
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
        min_distance=min(40.0, curr_dist),
        avg_distance=70.0,
        current_distance=curr_dist,
        relative_velocity=rel_vel,
        relative_acceleration=rel_acc,
        heading_difference=heading_diff,
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
# BehaviourGraphEngine Tests
# ======================================================================

class TestBehaviourGraphEngine:

    def test_create_graph(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001")

        graph = engine.create_graph(interaction, frame_number=1)
        assert graph.interaction_id == "INT-0001"
        assert graph.is_active is True
        assert graph.graph_id == "GRAPH-INT-0001"
        assert len(graph.nodes) == 0

    def test_update_graph_creates_node(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001", curr_dist=40.0)
        prims = [_make_primitive("CLOSE_INTERACTION")]

        graph = engine.update_graph(interaction, prims, [], frame_number=5)

        assert len(graph.nodes) >= 1
        assert graph.nodes[0].pattern_type == "INTERACTION_PATTERN"

    def test_pattern_transition_creates_edge(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001", rel_vel=-3.0, curr_dist=120.0)
        prims_approach = [_make_primitive("APPROACHING")]

        # Phase 1: APPROACH_PATTERN
        engine.update_graph(interaction, prims_approach, [], frame_number=5)

        # Phase 2: SEPARATION_PATTERN
        interaction.relative_velocity = 5.0
        prims_sep = [_make_primitive("MOVING_AWAY")]
        graph = engine.update_graph(interaction, prims_sep, [], frame_number=10)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

        edge = graph.edges[0]
        assert edge.from_pattern_type == "APPROACH_PATTERN"
        assert edge.to_pattern_type == "SEPARATION_PATTERN"
        assert edge.transition_frame == 10

    def test_finalize_graph(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001")
        engine.create_graph(interaction, frame_number=1)

        finalized = engine.finalize_graph("INT-0001", frame_number=50)
        assert finalized.is_active is False
        assert finalized.end_frame == 50

    def test_get_active_and_completed_graphs(self):
        engine = BehaviourGraphEngine()
        i1 = _make_interaction("INT-0001")
        i2 = _make_interaction("INT-0002")

        engine.create_graph(i1, frame_number=1)
        engine.create_graph(i2, frame_number=1)

        engine.finalize_graph("INT-0001", frame_number=20)

        active = engine.get_active_graphs()
        completed = engine.get_completed_graphs()

        assert len(active) == 1
        assert active[0].interaction_id == "INT-0002"

        assert len(completed) == 1
        assert completed[0].interaction_id == "INT-0001"

    def test_find_patterns(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001", curr_dist=40.0)
        prims = [_make_primitive("CLOSE_INTERACTION")]

        engine.update_graph(interaction, prims, [], frame_number=5)

        matches = engine.find_patterns(pattern_type="INTERACTION_PATTERN")
        assert len(matches) == 1

        no_matches = engine.find_patterns(pattern_type="NON_EXISTENT_PATTERN")
        assert len(no_matches) == 0

    def test_get_transition_statistics(self):
        engine = BehaviourGraphEngine()
        interaction = _make_interaction("INT-0001", rel_vel=-3.0, curr_dist=120.0)

        # Transition 1
        engine.update_graph(interaction, [_make_primitive("APPROACHING")], [], frame_number=5)
        # Transition 2
        interaction.relative_velocity = 5.0
        engine.update_graph(interaction, [_make_primitive("MOVING_AWAY")], [], frame_number=10)

        stats = engine.get_transition_statistics()
        assert stats["total_transitions"] == 1
        assert len(stats["transitions"]) == 1
        assert stats["transitions"][0]["from_pattern"] == "APPROACH_PATTERN"
        assert stats["transitions"][0]["to_pattern"] == "SEPARATION_PATTERN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

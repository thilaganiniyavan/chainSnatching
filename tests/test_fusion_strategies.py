"""Unit tests for the FusionStrategyEngine module.

Tests cover all 5 fusion strategies:
- weighted_confidence
- bayesian
- rule_based
- voting_based
- weighted_averaging
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.core.models.action_result import ActionResult
from src.behavior.fusion_strategies import FusionStrategyEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_graph() -> BehaviourGraph:
    return BehaviourGraph(
        graph_id="GRAPH-INT-001",
        interaction_id="INT-001",
        person_track_id=1,
        vehicle_track_id=2,
        nodes=[
            PatternNode(pattern_type="APPROACH_PATTERN", confidence=0.8),
            PatternNode(pattern_type="INTERACTION_PATTERN", confidence=0.9),
        ],
    )


def _make_action_result() -> ActionResult:
    return ActionResult(
        sequence_id="SEQ-INT-001-TRK-1",
        interaction_id="INT-001",
        track_id=1,
        predicted_action="Reaching",
        action_confidence=0.85,
    )


# ======================================================================
# FusionStrategyEngine Tests
# ======================================================================

class TestFusionStrategyEngine:

    def test_weighted_confidence_strategy(self):
        engine = FusionStrategyEngine(strategy="weighted_confidence")
        graph = _make_graph()
        actions = [_make_action_result()]

        b_conf, a_conf, f_conf = engine.fuse(graph, actions, motion_conf=0.80)
        assert b_conf == 0.85
        assert a_conf == 0.85
        assert 0.0 <= f_conf <= 1.0

    def test_bayesian_strategy(self):
        engine = FusionStrategyEngine(strategy="bayesian")
        graph = _make_graph()
        actions = [_make_action_result()]

        b_conf, a_conf, f_conf = engine.fuse(graph, actions)
        assert 0.0 <= f_conf <= 1.0

    def test_rule_based_strategy(self):
        engine = FusionStrategyEngine(strategy="rule_based")
        graph = _make_graph() # contains INTERACTION_PATTERN & APPROACH_PATTERN
        actions = [_make_action_result()] # contains Reaching (triggers bonus rule)

        b_conf, a_conf, f_conf = engine.fuse(graph, actions)
        # Should apply rule bonus (+0.15 for INTERACTION_PATTERN + Reaching)
        assert f_conf > 0.5 * (b_conf + a_conf)

    def test_voting_based_strategy(self):
        engine = FusionStrategyEngine(strategy="voting_based")
        graph = _make_graph()
        actions = [_make_action_result()]

        b_conf, a_conf, f_conf = engine.fuse(graph, actions)
        assert 0.0 <= f_conf <= 1.0

    def test_weighted_averaging_strategy(self):
        engine = FusionStrategyEngine(strategy="weighted_averaging")
        graph = _make_graph()
        actions = [_make_action_result()]

        b_conf, a_conf, f_conf = engine.fuse(graph, actions, motion_conf=0.80)
        expected_avg = round((b_conf + a_conf + 0.80) / 3.0, 4)
        assert f_conf == expected_avg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

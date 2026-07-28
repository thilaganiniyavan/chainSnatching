"""Unit tests for the BehaviourFusionEngine and FusionExplainer modules.

Tests cover:
- Multi-modal evidence fusion execution
- FusedInteraction creation and field structure
- Fusion APIs (fuse_interaction, update_fusion, finalize_fusion, get_completed_fusions)
- Human-readable forensic explanation generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.core.models.action_result import ActionResult
from src.core.models.fused_interaction import FusedInteraction
from src.behavior.behaviour_fusion_engine import BehaviourFusionEngine
from src.behavior.fusion_explainer import FusionExplainer


# ======================================================================
# Helpers
# ======================================================================

def _make_graph() -> BehaviourGraph:
    return BehaviourGraph(
        graph_id="GRAPH-INT-001",
        interaction_id="INT-001",
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        end_frame=10,
        nodes=[
            PatternNode(pattern_type="APPROACH_PATTERN", start_frame=1, end_frame=2, confidence=0.8),
            PatternNode(pattern_type="INTERACTION_PATTERN", start_frame=5, end_frame=6, confidence=0.9),
        ],
    )


def _make_action_result() -> ActionResult:
    return ActionResult(
        sequence_id="SEQ-INT-001-TRK-1",
        interaction_id="INT-001",
        track_id=1,
        predicted_action="Reaching",
        action_confidence=0.85,
        metadata={"frame_index": 5},
    )


# ======================================================================
# BehaviourFusionEngine Tests
# ======================================================================

class TestBehaviourFusionEngine:

    def test_fuse_interaction(self):
        engine = BehaviourFusionEngine(fusion_strategy="weighted_confidence")
        graph = _make_graph()
        actions = [_make_action_result()]

        fused = engine.fuse_interaction(graph, actions)
        assert isinstance(fused, FusedInteraction)
        assert fused.interaction_id == "INT-001"
        assert fused.person_track_id == 1
        assert fused.vehicle_track_id == 2
        assert fused.behaviour_patterns == ["APPROACH_PATTERN", "INTERACTION_PATTERN"]
        assert fused.behaviour_confidence == 0.85
        assert fused.action_confidence == 0.85
        assert fused.fusion_confidence > 0.0
        assert len(fused.explanation_text) > 0

    def test_get_completed_fusions(self):
        engine = BehaviourFusionEngine()
        graph = _make_graph()
        actions = [_make_action_result()]

        engine.fuse_interaction(graph, actions)
        completed = engine.get_completed_fusions()
        assert len(completed) == 1
        assert completed[0].fusion_id == "FUSED-INT-001"

    def test_fusion_explainer(self):
        explainer = FusionExplainer()
        fusion = FusedInteraction(
            fusion_id="FUSED-INT-001",
            interaction_id="INT-001",
            person_track_id=1,
            vehicle_track_id=2,
            duration_seconds=2.1,
            behaviour_patterns=["APPROACH_PATTERN", "INTERACTION_PATTERN"],
            action_timeline=[{"action_label": "Reaching", "action_confidence": 0.91}],
            spatial_evidence={"min_distance_px": 45.0},
            fusion_confidence=0.88,
        )

        text = explainer.generate_explanation(fusion)
        assert "Vehicle (Track 2)" in text
        assert "Person (Track 1)" in text
        assert "APPROACH_PATTERN -> INTERACTION_PATTERN" in text
        assert "Reaching" in text
        assert "88%" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

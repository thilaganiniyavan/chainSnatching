"""Unit tests for the FusionTemporalAligner module.

Tests cover:
- Multi-modal evidence timeline alignment across timestamps
- Merging Behaviour Graph nodes and Action Results
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.core.models.action_result import ActionResult
from src.behavior.fusion_temporal_aligner import FusionTemporalAligner


# ======================================================================
# FusionTemporalAligner Tests
# ======================================================================

class TestFusionTemporalAligner:

    def test_align_streams(self):
        aligner = FusionTemporalAligner()
        graph = BehaviourGraph(
            graph_id="GRAPH-001",
            interaction_id="INT-001",
            start_frame=1,
            nodes=[
                PatternNode(pattern_type="APPROACH_PATTERN", start_frame=1, end_frame=2, confidence=0.8),
                PatternNode(pattern_type="INTERACTION_PATTERN", start_frame=5, end_frame=6, confidence=0.9),
            ],
        )

        act = ActionResult(
            sequence_id="SEQ-INT-001",
            interaction_id="INT-001",
            predicted_action="Reaching",
            action_confidence=0.85,
            metadata={"frame_indices": [5, 6]},
        )

        timeline = aligner.align_streams(graph, [act], fps=30.0)
        assert len(timeline) >= 2
        # Frame 5 should contain both pattern and action label
        frame5_event = next(e for e in timeline if e["frame"] == 5)
        assert frame5_event["behaviour_pattern"] == "INTERACTION_PATTERN"
        assert frame5_event["action_label"] == "Reaching"
        assert frame5_event["action_confidence"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

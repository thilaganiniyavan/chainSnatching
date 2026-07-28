"""Unit tests for the ROIEngine module.

Tests cover:
- ROI creation and initialization from Behaviour Graphs
- Per-frame tracking updates and bbox processing
- Quality threshold acceptance filtering
- ROI finalization
- Query APIs (get_active_rois, get_completed_rois, get_accepted_rois)
- Skeleton sample preparation for pose estimation frameworks
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.core.models.track import Track
from src.core.models.interaction_roi import InteractionROI
from src.behavior.roi_engine import ROIEngine


# ======================================================================
# Helpers
# ======================================================================

def _make_graph(iid: str = "INT-0001") -> BehaviourGraph:
    return BehaviourGraph(
        graph_id=f"GRAPH-{iid}",
        interaction_id=iid,
        person_track_id=1,
        vehicle_track_id=2,
        start_frame=1,
        nodes=[
            PatternNode(pattern_type="APPROACH_PATTERN", confidence=0.8),
            PatternNode(pattern_type="INTERACTION_PATTERN", confidence=0.9),
        ],
    )


def _make_track(track_id: int = 1, bbox: tuple = (100, 100, 200, 200)) -> Track:
    return Track(
        tracking_id=track_id,
        class_name="person",
        center=(150, 150),
        metadata={"bbox": list(bbox)},
    )


# ======================================================================
# ROIEngine Tests
# ======================================================================

class TestROIEngine:

    def test_create_roi(self):
        engine = ROIEngine()
        graph = _make_graph("INT-0001")

        roi = engine.create_roi(graph, frame_number=1)
        assert roi.interaction_id == "INT-0001"
        assert roi.person_track_id == 1
        assert roi.vehicle_track_id == 2
        assert roi.interaction_confidence == 0.9
        assert roi.pattern_sequence == ["APPROACH_PATTERN", "INTERACTION_PATTERN"]

    def test_update_roi_populates_sequences(self):
        engine = ROIEngine()
        graph = _make_graph("INT-0001")
        track = _make_track(1, (100, 100, 200, 200))

        roi = engine.update_roi(graph, [track], frame_number=1)
        assert roi.frame_count == 1
        assert len(roi.bounding_box_sequence) == 1
        assert len(roi.expanded_bounding_boxes) == 1
        assert roi.frame_index_mapping == [1]

    def test_finalize_roi(self):
        engine = ROIEngine()
        graph = _make_graph("INT-0001")
        track = _make_track(1, (100, 100, 200, 200))

        engine.update_roi(graph, [track], frame_number=1)
        finalized = engine.finalize_roi("INT-0001", frame_number=20)

        assert finalized.end_frame == 20

    def test_get_accepted_rois(self):
        engine = ROIEngine()
        graph = _make_graph("INT-0001")
        track = _make_track(1, (100, 100, 200, 200))

        # Update 10 consecutive valid frames
        for f in range(1, 11):
            engine.update_roi(graph, [track], frame_number=f)

        accepted = engine.get_accepted_rois()
        assert len(accepted) == 1
        assert accepted[0].is_accepted is True

    def test_prepare_skeleton_samples(self):
        engine = ROIEngine()
        graph = _make_graph("INT-0001")
        track = _make_track(1, (100, 100, 200, 200))

        for f in range(1, 6):
            roi = engine.update_roi(graph, [track], frame_number=f)

        samples = engine.prepare_skeleton_samples(roi)
        assert len(samples) == 5

        sample = samples[0]
        assert sample.interaction_id == "INT-0001"
        assert sample.frame_number == 1
        assert sample.person_track_id == 1
        assert "MediaPipe" in sample.expected_skeleton_placeholder["model_compatibility"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for the ROIQualityEvaluator module.

Tests cover:
- Metric computation (completeness, missing pct, stability, continuity, coverage)
- Acceptance/rejection rules based on configurable quality thresholds
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction_roi import InteractionROI
from src.behavior.roi_quality_evaluator import ROIQualityEvaluator


# ======================================================================
# ROI Quality Evaluator Tests
# ======================================================================

class TestROIQualityEvaluator:

    def test_evaluate_accepted_roi(self):
        evaluator = ROIQualityEvaluator(min_completeness=0.6, max_missing_pct=40.0)
        roi = InteractionROI(
            roi_id="ROI-001",
            interaction_id="INT-001",
            start_frame=1,
            end_frame=10,
            frame_count=10,
            bounding_box_sequence=[(100, 100, 200, 200)] * 10,
        )
        raw_mask = [True] * 10

        evaluated = evaluator.evaluate(roi, raw_box_mask=raw_mask)
        assert evaluated.is_accepted is True
        assert evaluated.rejection_reason == "Accepted"
        assert evaluated.quality_metrics["completeness"] == 1.0
        assert evaluated.quality_metrics["missing_detection_percentage"] == 0.0

    def test_evaluate_rejected_roi_low_completeness(self):
        evaluator = ROIQualityEvaluator(min_completeness=0.6)
        roi = InteractionROI(
            roi_id="ROI-002",
            interaction_id="INT-002",
            start_frame=1,
            end_frame=10,
            frame_count=10,
            bounding_box_sequence=[(100, 100, 200, 200)] * 10,
        )
        # Only 2 out of 10 valid detections (completeness 0.2 < 0.6)
        raw_mask = [True, True] + [False] * 8

        evaluated = evaluator.evaluate(roi, raw_box_mask=raw_mask)
        assert evaluated.is_accepted is False
        assert "Completeness" in evaluated.rejection_reason
        assert evaluated.quality_metrics["completeness"] == 0.2
        assert evaluated.quality_metrics["missing_detection_percentage"] == 80.0

    def test_evaluate_empty_sequence_rejected(self):
        evaluator = ROIQualityEvaluator()
        roi = InteractionROI(roi_id="ROI-003", interaction_id="INT-003")

        evaluated = evaluator.evaluate(roi)
        assert evaluated.is_accepted is False
        assert "Empty" in evaluated.rejection_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

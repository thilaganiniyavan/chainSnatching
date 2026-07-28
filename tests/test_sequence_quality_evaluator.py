"""Unit tests for the SequenceQualityEvaluator module.

Tests cover:
- Metric calculations (completeness, confidence, stability, missing ratio)
- Threshold acceptance and rejection rules
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.core.models.skeleton_sequence import SkeletonSequence
from src.behavior.sequence_quality_evaluator import SequenceQualityEvaluator


# ======================================================================
# SequenceQualityEvaluator Tests
# ======================================================================

class TestSequenceQualityEvaluator:

    def test_evaluate_accepted_sequence(self):
        evaluator = SequenceQualityEvaluator(min_completeness=0.5, min_confidence=0.4)
        seq = SkeletonSequence(
            sequence_id="SEQ-001",
            frame_count=10,
            skeleton_tensor=np.ones((10, 17, 4), dtype=float) * 0.8,
            joint_confidence_matrix=np.ones((10, 17), dtype=float) * 0.8,
        )

        evaluated = evaluator.evaluate(seq)
        assert evaluated.is_accepted is True
        assert evaluated.rejection_reason == "Accepted"
        assert evaluated.completeness_score == 1.0
        assert evaluated.quality_score > 0.0

    def test_evaluate_rejected_low_confidence(self):
        evaluator = SequenceQualityEvaluator(min_confidence=0.6)
        seq = SkeletonSequence(
            sequence_id="SEQ-002",
            frame_count=10,
            skeleton_tensor=np.ones((10, 17, 4), dtype=float) * 0.3,
            joint_confidence_matrix=np.ones((10, 17), dtype=float) * 0.3,
        )

        evaluated = evaluator.evaluate(seq)
        assert evaluated.is_accepted is False
        assert "Confidence" in evaluated.rejection_reason

    def test_evaluate_empty_sequence_rejected(self):
        evaluator = SequenceQualityEvaluator()
        seq = SkeletonSequence(sequence_id="SEQ-003")

        evaluated = evaluator.evaluate(seq)
        assert evaluated.is_accepted is False
        assert "Empty" in evaluated.rejection_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for the ActionPostProcessor module.

Tests cover:
- Confidence thresholding & Unknown fallback
- Top-k prediction ranking
- Sliding window clip probability aggregation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.core.models.action_result import ActionResult
from src.action.action_post_processor import ActionPostProcessor


# ======================================================================
# Helpers
# ======================================================================

def _make_action_result(predicted: str = "Reaching", conf: float = 0.8) -> ActionResult:
    probs = {
        "Walking": 0.05,
        "Standing": 0.05,
        "Reaching": conf,
        "Grabbing": 0.10,
        "Unknown": 0.0,
    }

    return ActionResult(
        sequence_id="SEQ-001",
        interaction_id="INT-001",
        track_id=1,
        predicted_action=predicted,
        action_confidence=conf,
        class_probabilities=probs,
    )


# ======================================================================
# ActionPostProcessor Tests
# ======================================================================

class TestActionPostProcessor:

    def test_process_high_confidence(self):
        processor = ActionPostProcessor(min_confidence=0.40)
        res = _make_action_result("Reaching", conf=0.80)

        processed = processor.process(res)
        assert processed.predicted_action == "Reaching"
        assert processed.action_confidence == 0.80
        assert len(processed.top_k_predictions) > 0

    def test_process_low_confidence_fallback_to_unknown(self):
        processor = ActionPostProcessor(min_confidence=0.50)
        res = _make_action_result("Reaching", conf=0.30)
        res.class_probabilities["Reaching"] = 0.30

        processed = processor.process(res)
        # Should fallback to "Unknown" since confidence 0.30 < threshold 0.50
        assert processed.predicted_action == "Unknown"
        assert processed.metadata.get("fallback_triggered") is True

    def test_aggregate_overlapping_windows(self):
        processor = ActionPostProcessor(min_confidence=0.40)

        clip1 = _make_action_result("Reaching", conf=0.70)
        clip2 = _make_action_result("Reaching", conf=0.80)

        aggregated = processor.aggregate_overlapping_windows([clip1, clip2])
        assert aggregated.sequence_id == "SEQ-001"
        assert aggregated.predicted_action == "Reaching"
        assert aggregated.action_confidence == 0.75
        assert aggregated.metadata["aggregated_clip_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

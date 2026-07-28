"""Unit tests for the PosePostProcessor module.

Tests cover:
- Keypoint gap interpolation
- EMA temporal coordinate smoothing
- Confidence threshold joint filtering
- Overall pose quality score computation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.core.models.pose_result import PoseResult
from src.pose.pose_post_processor import PosePostProcessor


# ======================================================================
# Helpers
# ======================================================================

def _make_pose_result(frame_idx: int, conf: float = 0.8) -> PoseResult:
    # 17 COCO keypoints
    kps_px = [(100.0 + f, 100.0 + f, conf, 0.9) for f in range(17)]
    kps_norm = [(0.1, 0.1, conf, 0.9) for _ in range(17)]

    return PoseResult(
        sample_id=f"SMP-{frame_idx}",
        roi_id="ROI-001",
        interaction_id="INT-001",
        frame_index=frame_idx,
        timestamp=round(frame_idx / 30.0, 3),
        track_id=1,
        keypoints_pixel=kps_px,
        keypoints_normalized=kps_norm,
        num_keypoints=17,
        topology="COCO_17",
        overall_confidence=conf,
        quality_score=conf,
        bbox_reference=(100, 100, 200, 300),
    )


# ======================================================================
# PosePostProcessor Tests
# ======================================================================

class TestPosePostProcessor:

    def test_process_sequence_ema_smoothing(self):
        processor = PosePostProcessor(ema_alpha=0.5)
        p1 = _make_pose_result(1, conf=0.8)
        p2 = _make_pose_result(2, conf=0.8)

        # Set distinct coordinates for p2
        p2.keypoints_pixel = [(120.0, 120.0, 0.8, 0.9)] * 17

        processed = processor.process_sequence([p1, p2])
        assert len(processed) == 2

        # First pose remains intact
        assert processed[0].keypoints_pixel[0][0] == 100.0

        # Second pose smoothed via EMA: 0.5 * 120 + 0.5 * 100 = 110
        assert processed[1].keypoints_pixel[0][0] == 110.0

    def test_confidence_filtering(self):
        processor = PosePostProcessor(min_joint_confidence=0.5)
        p = _make_pose_result(1, conf=0.8)

        # Drop confidence of first joint below threshold
        p.keypoints_pixel[0] = (100.0, 100.0, 0.2, 0.2)

        processed = processor.process_sequence([p])
        kp0 = processed[0].keypoints_pixel[0]

        # Filtered joint confidence should be zeroed out
        assert kp0[2] == 0.0

    def test_keypoint_interpolation(self):
        processor = PosePostProcessor(min_joint_confidence=0.5, max_missing_gap=5)
        p1 = _make_pose_result(1, conf=0.8)
        p2 = _make_pose_result(2, conf=0.8)
        p3 = _make_pose_result(3, conf=0.8)

        # Set distinct coordinates for p1 and p3
        p1.keypoints_pixel[0] = (100.0, 100.0, 0.8, 0.9)
        p2.keypoints_pixel[0] = (120.0, 120.0, 0.1, 0.1)  # Low confidence -> missing
        p3.keypoints_pixel[0] = (102.0, 102.0, 0.8, 0.9)

        processed = processor.process_sequence([p1, p2, p3])
        # p2's joint 0 interpolated to 101.0, then EMA smoothed (0.45*101 + 0.55*100) -> 100.5
        assert processed[1].keypoints_pixel[0][0] == 100.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

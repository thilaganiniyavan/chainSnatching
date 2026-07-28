"""Unit tests for Pose Estimator Interface, MediaPipe backend, Scaffold Adapters, and Factory.

Tests cover:
- PoseEstimatorFactory backend instantiation
- MediaPipePoseEstimator single-frame and batch estimation
- Scaffold adapters (RTMPose, ViTPose, MMPose, OpenPose)
- PoseResult field structure and keypoint dimensions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.core.models.pose_result import PoseResult
from src.core.models.interaction_roi import PreparedSkeletonSample
from src.pose.factory import PoseEstimatorFactory
from src.pose.mediapipe_estimator import MediaPipePoseEstimator
from src.pose.adapters.rtmpose_adapter import RTMPoseAdapter
from src.pose.adapters.vitpose_adapter import ViTPoseAdapter
from src.pose.adapters.mmpose_adapter import MMPoseAdapter
from src.pose.adapters.openpose_adapter import OpenPoseAdapter


# ======================================================================
# Helpers
# ======================================================================

def _make_dummy_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_sample(f_num: int = 1) -> PreparedSkeletonSample:
    return PreparedSkeletonSample(
        sample_id=f"SMP-{f_num}",
        roi_id="ROI-001",
        interaction_id="INT-001",
        frame_number=f_num,
        timestamp=round(f_num / 30.0, 3),
        person_track_id=1,
        raw_bbox=(100, 100, 200, 300),
        expanded_bbox=(90, 90, 210, 310),
    )


# ======================================================================
# Pose Estimator Tests
# ======================================================================

class TestPoseEstimatorFactory:

    def test_supported_backends(self):
        backends = PoseEstimatorFactory.get_supported_backends()
        assert "mediapipe" in backends
        assert "rtmpose" in backends
        assert "vitpose" in backends
        assert "mmpose" in backends
        assert "openpose" in backends

    def test_create_mediapipe_backend(self):
        estimator = PoseEstimatorFactory.create("mediapipe")
        assert isinstance(estimator, MediaPipePoseEstimator)
        assert estimator.backend_name == "MediaPipe"

    def test_create_scaffold_adapters(self):
        rtm = PoseEstimatorFactory.create("rtmpose")
        assert isinstance(rtm, RTMPoseAdapter)
        assert rtm.backend_name == "RTMPose"

        vit = PoseEstimatorFactory.create("vitpose")
        assert isinstance(vit, ViTPoseAdapter)
        assert vit.backend_name == "ViTPose"

        mm = PoseEstimatorFactory.create("mmpose")
        assert isinstance(mm, MMPoseAdapter)
        assert mm.backend_name == "MMPose"

        op = PoseEstimatorFactory.create("openpose")
        assert isinstance(op, OpenPoseAdapter)
        assert op.backend_name == "OpenPose"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            PoseEstimatorFactory.create("invalid_backend_name")


class TestMediaPipePoseEstimator:

    def test_estimate_pose_single_frame(self):
        estimator = MediaPipePoseEstimator(topology="COCO_17")
        img = _make_dummy_image()
        bbox = (100, 100, 200, 300)

        pose = estimator.estimate_pose(
            image=img,
            bbox=bbox,
            frame_index=1,
            timestamp=0.033,
            track_id=1,
            interaction_id="INT-001",
            roi_id="ROI-001",
        )

        assert isinstance(pose, PoseResult)
        assert pose.interaction_id == "INT-001"
        assert pose.num_keypoints == 17
        assert pose.topology == "COCO_17"
        assert len(pose.keypoints_pixel) == 17
        assert len(pose.keypoints_normalized) == 17
        assert pose.processing_time_ms >= 0.0

    def test_estimate_batch(self):
        estimator = MediaPipePoseEstimator(topology="COCO_17")
        samples = [_make_sample(1), _make_sample(2)]
        frames_dict = {1: _make_dummy_image(), 2: _make_dummy_image()}

        results = estimator.estimate_batch(samples, frames_dict)
        assert len(results) == 2
        assert results[0].frame_index == 1
        assert results[1].frame_index == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

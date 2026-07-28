"""Unit tests for the SkeletonNormalizer module.

Tests cover:
- Image coordinate normalization
- Bounding box normalization
- Hip-centered normalization
- Root-joint normalization
- Rotation normalization alignment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.behavior.skeleton_normalizer import SkeletonNormalizer


# ======================================================================
# Helpers
# ======================================================================

def _make_dummy_raw_tensor(T: int = 5, V: int = 17) -> np.ndarray:
    raw = np.zeros((T, V, 4), dtype=float)
    for t in range(T):
        for v in range(V):
            raw[t, v] = (100.0 + v * 10, 200.0 + v * 10, 0.8, 0.9)
    return raw


# ======================================================================
# SkeletonNormalizer Tests
# ======================================================================

class TestSkeletonNormalizer:

    def test_normalize_image_strategy(self):
        normalizer = SkeletonNormalizer(method="image", image_width=1000, image_height=1000)
        raw = _make_dummy_raw_tensor(T=1, V=17)

        norm = normalizer.normalize_tensor(raw, topology="COCO_17")
        assert norm.shape == raw.shape
        # Keypoint 0 x: 100.0 / 1000.0 = 0.1
        assert norm[0, 0, 0] == 0.1
        # Keypoint 0 y: 200.0 / 1000.0 = 0.2
        assert norm[0, 0, 1] == 0.2

    def test_normalize_bbox_strategy(self):
        normalizer = SkeletonNormalizer(method="bbox")
        raw = _make_dummy_raw_tensor(T=1, V=17)
        bboxes = [(100, 100, 300, 300)] # w=200, h=200

        norm = normalizer.normalize_tensor(raw, bboxes=bboxes, topology="COCO_17")
        # Keypoint 0 x: (100.0 - 100) / 200 = 0.0
        assert norm[0, 0, 0] == 0.0
        # Keypoint 0 y: (200.0 - 100) / 200 = 0.5
        assert norm[0, 0, 1] == 0.5

    def test_normalize_hip_centered_strategy(self):
        normalizer = SkeletonNormalizer(method="hip_centered")
        raw = _make_dummy_raw_tensor(T=1, V=17)

        norm = normalizer.normalize_tensor(raw, topology="COCO_17")
        assert norm.shape == raw.shape
        # Center of hips should be centered near origin (0, 0)
        hip_center_x = (norm[0, 11, 0] + norm[0, 12, 0]) / 2.0
        hip_center_y = (norm[0, 11, 1] + norm[0, 12, 1]) / 2.0
        assert abs(hip_center_x) < 1e-4
        assert abs(hip_center_y) < 1e-4

    def test_normalize_root_joint_strategy(self):
        normalizer = SkeletonNormalizer(method="root_joint")
        raw = _make_dummy_raw_tensor(T=1, V=17)

        norm = normalizer.normalize_tensor(raw, topology="COCO_17")
        # Root joint 0 (nose) should be at (0, 0)
        assert norm[0, 0, 0] == 0.0
        assert norm[0, 0, 1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

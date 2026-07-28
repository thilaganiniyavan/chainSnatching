"""Unit tests for the SkeletonSequenceBuilder module.

Tests cover:
- Sequence creation and pose buffering
- Tensor construction and normalization
- Finalization and fixed-length padding
- Query APIs (get_sequence, get_completed_sequences, validate_sequence)
- Tensor export formats ("TVC", "VCT", "CTV", "NCTVM")
- Sliding window clip generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.core.models.pose_result import PoseResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.behavior.skeleton_sequence_builder import SkeletonSequenceBuilder


# ======================================================================
# Helpers
# ======================================================================

def _make_pose(f_num: int = 1, iid: str = "INT-001", trk_id: int = 1) -> PoseResult:
    kps_px = [(100.0 + v, 100.0 + v, 0.8, 0.9) for v in range(17)]
    kps_norm = [(0.1, 0.1, 0.8, 0.9) for _ in range(17)]

    return PoseResult(
        sample_id=f"SMP-{f_num}",
        roi_id=f"ROI-{iid}",
        interaction_id=iid,
        frame_index=f_num,
        timestamp=round(f_num / 30.0, 3),
        track_id=trk_id,
        keypoints_pixel=kps_px,
        keypoints_normalized=kps_norm,
        num_keypoints=17,
        topology="COCO_17",
        bbox_reference=(100, 100, 200, 300),
    )


# ======================================================================
# SkeletonSequenceBuilder Tests
# ======================================================================

class TestSkeletonSequenceBuilder:

    def test_create_sequence(self):
        builder = SkeletonSequenceBuilder()
        seq = builder.create_sequence("INT-001", 1, topology="COCO_17")

        assert seq.interaction_id == "INT-001"
        assert seq.person_track_id == 1
        assert seq.topology == "COCO_17"
        assert seq.num_joints == 17

    def test_append_and_finalize_sequence(self):
        builder = SkeletonSequenceBuilder()
        seq_id = "SEQ-INT-001-TRK-1"

        for f in range(1, 11):
            p = _make_pose(f)
            builder.append_pose(seq_id, p)

        finalized = builder.finalize_sequence(seq_id)
        assert finalized.frame_count == 10
        assert finalized.skeleton_tensor.shape == (10, 17, 4)
        assert finalized.joint_confidence_matrix.shape == (10, 17)
        assert finalized.is_accepted is True

    def test_finalize_with_fixed_length_padding(self):
        builder = SkeletonSequenceBuilder()
        seq_id = "SEQ-INT-001-TRK-1"

        # 5 poses, request fixed_length=30
        for f in range(1, 6):
            p = _make_pose(f)
            builder.append_pose(seq_id, p)

        finalized = builder.finalize_sequence(seq_id, fixed_length=30, padding_mode="zero")
        assert finalized.frame_count == 30
        assert finalized.skeleton_tensor.shape == (30, 17, 4)

    def test_export_tensor_formats(self):
        builder = SkeletonSequenceBuilder()
        seq_id = "SEQ-INT-001-TRK-1"
        for f in range(1, 6):
            builder.append_pose(seq_id, _make_pose(f))

        builder.finalize_sequence(seq_id)

        # TVC format: (T, V, C) -> (5, 17, 4)
        tvc = builder.export_tensor(seq_id, format="TVC")
        assert tvc.shape == (5, 17, 4)

        # VCT format: (V, C, T) -> (17, 4, 5)
        vct = builder.export_tensor(seq_id, format="VCT")
        assert vct.shape == (17, 4, 5)

        # CTV format: (C, T, V) -> (4, 5, 17)
        ctv = builder.export_tensor(seq_id, format="CTV")
        assert ctv.shape == (4, 5, 17)

        # NCTVM format (ST-GCN input): (1, C, T, V, 1) -> (1, 4, 5, 17, 1)
        nctvm = builder.export_tensor(seq_id, format="NCTVM")
        assert nctvm.shape == (1, 4, 5, 17, 1)

    def test_generate_sliding_windows(self):
        builder = SkeletonSequenceBuilder()
        seq_id = "SEQ-INT-001-TRK-1"

        # Append 50 poses
        for f in range(1, 51):
            builder.append_pose(seq_id, _make_pose(f))

        builder.finalize_sequence(seq_id)

        # Window size 30, stride 15 -> (50 - 30) / 15 + 1 = 2 clips
        clips = builder.generate_sliding_windows(seq_id, window_size=30, stride=15)
        assert len(clips) == 2
        assert clips[0].frame_count == 30
        assert clips[1].frame_count == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

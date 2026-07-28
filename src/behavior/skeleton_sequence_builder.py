"""Skeleton Sequence Builder Engine.

Consumes chronological frame-wise PoseResult objects to build temporally ordered,
spatially normalized, and validated SkeletonSequence objects.

Provides clean API suite:
- create_sequence()
- append_pose()
- finalize_sequence()
- get_sequence()
- get_completed_sequences()
- validate_sequence()
- export_tensor()
- generate_sliding_windows()
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.core.models.pose_result import PoseResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.behavior.skeleton_normalizer import SkeletonNormalizer
from src.behavior.sequence_quality_evaluator import SequenceQualityEvaluator


class SkeletonSequenceBuilder:
    """Manages creation, pose buffering, spatial normalization, tensor formatting,
    quality evaluation, and sliding window clip generation for SkeletonSequences.

    Args:
        normalizer: Custom SkeletonNormalizer instance.
        evaluator: Custom SequenceQualityEvaluator instance.
        fps: Video frame rate.
    """

    def __init__(
        self,
        normalizer: SkeletonNormalizer | None = None,
        evaluator: SequenceQualityEvaluator | None = None,
        fps: float = 30.0,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.normalizer = normalizer if normalizer is not None else SkeletonNormalizer()
        self.evaluator = evaluator if evaluator is not None else SequenceQualityEvaluator()

        # Storage: sequence_id -> SkeletonSequence
        self._sequences: dict[str, SkeletonSequence] = {}
        # Buffer: sequence_id -> list[PoseResult]
        self._pose_buffers: dict[str, list[PoseResult]] = {}

    def create_sequence(
        self,
        interaction_id: str,
        person_track_id: int,
        topology: str = "COCO_17",
    ) -> SkeletonSequence:
        """Initialize a new SkeletonSequence."""
        seq_id = f"SEQ-{interaction_id}-TRK-{person_track_id}"
        num_joints = 17 if topology.upper() == "COCO_17" else 33

        sequence = SkeletonSequence(
            sequence_id=seq_id,
            interaction_id=interaction_id,
            person_track_id=person_track_id,
            topology=topology.upper(),
            num_joints=num_joints,
            normalization_method=self.normalizer.method,
        )

        self._sequences[seq_id] = sequence
        self._pose_buffers[seq_id] = []
        return sequence

    def append_pose(
        self,
        sequence_id: str,
        pose_result: PoseResult,
    ) -> None:
        """Append a frame PoseResult to an active sequence buffer."""
        if sequence_id not in self._sequences:
            self.create_sequence(
                interaction_id=pose_result.interaction_id,
                person_track_id=pose_result.track_id,
                topology=pose_result.topology,
            )

        self._pose_buffers[sequence_id].append(pose_result)

    def finalize_sequence(
        self,
        sequence_id: str,
        fixed_length: Optional[int] = None,
        padding_mode: str = "zero",
    ) -> SkeletonSequence:
        """Construct raw tensor, apply normalization, quality evaluation, and finalization.

        Args:
            sequence_id: Identifier of sequence.
            fixed_length: Optional target frame length T (e.g. 30 or 60).
            padding_mode: Padding strategy if length < fixed_length ("zero" or "repeat").

        Returns:
            The finalized :class:`SkeletonSequence` object.
        """
        seq = self._sequences.get(sequence_id)
        poses = self._pose_buffers.get(sequence_id, [])

        if seq is None or not poses:
            return seq if seq is not None else SkeletonSequence(sequence_id=sequence_id)

        # Sort poses by frame index
        poses = sorted(poses, key=lambda p: p.frame_index)

        num_joints = seq.num_joints
        n_frames = len(poses)

        raw_tensor = np.zeros((n_frames, num_joints, 4), dtype=float)
        conf_matrix = np.zeros((n_frames, num_joints), dtype=float)
        vis_matrix = np.zeros((n_frames, num_joints), dtype=float)

        timestamps: list[float] = []
        frame_indices: list[int] = []
        bboxes: list[tuple[int, int, int, int]] = []

        for t, p in enumerate(poses):
            timestamps.append(p.timestamp)
            frame_indices.append(p.frame_index)
            bboxes.append(p.bbox_reference)

            for v in range(num_joints):
                if v < len(p.keypoints_pixel):
                    kp = p.keypoints_pixel[v] # (x, y, conf, vis)
                    raw_tensor[t, v] = kp
                    conf_matrix[t, v] = kp[2]
                    vis_matrix[t, v] = kp[3]

        # 1. Apply spatial normalization
        norm_tensor = self.normalizer.normalize_tensor(
            raw_tensor, bboxes=bboxes, topology=seq.topology
        )

        # 2. Fixed-length padding or truncation if requested
        if fixed_length is not None and fixed_length > 0:
            norm_tensor, conf_matrix, vis_matrix = self._adjust_sequence_length(
                norm_tensor, conf_matrix, vis_matrix, fixed_length, padding_mode
            )
            n_frames = fixed_length

        seq.start_frame = frame_indices[0] if frame_indices else 0
        seq.end_frame = frame_indices[-1] if frame_indices else 0
        seq.frame_count = n_frames
        seq.duration_seconds = round(n_frames / self.fps, 3)

        seq.skeleton_tensor = norm_tensor
        seq.joint_confidence_matrix = conf_matrix
        seq.visibility_matrix = vis_matrix
        seq.timestamps = timestamps
        seq.frame_indices = frame_indices

        # 3. Evaluate quality
        self.evaluator.evaluate(seq)

        return seq

    def get_sequence(self, sequence_id: str) -> Optional[SkeletonSequence]:
        """Return SkeletonSequence by ID."""
        return self._sequences.get(sequence_id)

    def get_completed_sequences(self) -> list[SkeletonSequence]:
        """Return all finalized sequences."""
        return list(self._sequences.values())

    def validate_sequence(self, sequence: SkeletonSequence) -> tuple[bool, str]:
        """Validate sequence structure and non-empty tensor data."""
        if sequence.skeleton_tensor.size == 0:
            return False, "Empty skeleton tensor"
        if sequence.frame_count < 3:
            return False, "Frame count < 3 frames minimum requirement"
        return sequence.is_accepted, sequence.rejection_reason

    def export_tensor(
        self,
        sequence_id: str,
        format: str = "TVC",
    ) -> np.ndarray:
        """Export sequence skeleton tensor in requested shape format.

        Supported formats:
        - "TVC": shape (T, V, C) -> default
        - "VCT": shape (V, C, T)
        - "CTV": shape (C, T, V)
        - "NCTVM": shape (1, C, T, V, 1) -> ST-GCN standard input format
        """
        seq = self._sequences.get(sequence_id)
        if seq is None or seq.skeleton_tensor.size == 0:
            return np.zeros((0, 17, 4), dtype=float)

        tensor = seq.skeleton_tensor # (T, V, C)
        fmt = format.upper().strip()

        if fmt == "TVC":
            return tensor
        elif fmt == "VCT":
            return np.transpose(tensor, (1, 2, 0))
        elif fmt == "CTV":
            return np.transpose(tensor, (2, 0, 1))
        elif fmt == "NCTVM":
            ctv = np.transpose(tensor, (2, 0, 1)) # (C, T, V)
            return ctv[np.newaxis, :, :, :, np.newaxis] # (1, C, T, V, 1)
        else:
            return tensor

    def generate_sliding_windows(
        self,
        sequence_id: str,
        window_size: int = 30,
        stride: int = 15,
    ) -> list[SkeletonSequence]:
        """Generate sliding window clip sequences from a long interaction sequence."""
        parent_seq = self._sequences.get(sequence_id)
        if parent_seq is None or parent_seq.skeleton_tensor.shape[0] < window_size:
            return []

        T = parent_seq.skeleton_tensor.shape[0]
        clips: list[SkeletonSequence] = []

        clip_idx = 0
        for start_t in range(0, T - window_size + 1, stride):
            end_t = start_t + window_size
            clip_seq_id = f"{sequence_id}-CLIP-{clip_idx}"

            clip_tensor = parent_seq.skeleton_tensor[start_t:end_t]
            clip_conf = parent_seq.joint_confidence_matrix[start_t:end_t]
            clip_vis = parent_seq.visibility_matrix[start_t:end_t]

            clip_seq = SkeletonSequence(
                sequence_id=clip_seq_id,
                interaction_id=parent_seq.interaction_id,
                person_track_id=parent_seq.person_track_id,
                start_frame=parent_seq.frame_indices[start_t] if start_t < len(parent_seq.frame_indices) else start_t,
                end_frame=parent_seq.frame_indices[end_t - 1] if end_t - 1 < len(parent_seq.frame_indices) else end_t,
                frame_count=window_size,
                duration_seconds=round(window_size / self.fps, 3),
                topology=parent_seq.topology,
                num_joints=parent_seq.num_joints,
                skeleton_tensor=clip_tensor,
                joint_confidence_matrix=clip_conf,
                visibility_matrix=clip_vis,
                timestamps=parent_seq.timestamps[start_t:end_t],
                frame_indices=parent_seq.frame_indices[start_t:end_t],
                normalization_method=parent_seq.normalization_method,
            )

            self.evaluator.evaluate(clip_seq)
            clips.append(clip_seq)
            clip_idx += 1

        return clips

    @staticmethod
    def _adjust_sequence_length(
        tensor: np.ndarray,
        conf: np.ndarray,
        vis: np.ndarray,
        target_len: int,
        padding_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Truncate or pad tensor to fixed target_len frames."""
        T, V, C = tensor.shape
        if T == target_len:
            return tensor, conf, vis

        if T > target_len:
            # Truncate
            return tensor[:target_len], conf[:target_len], vis[:target_len]

        # Pad T < target_len
        pad_len = target_len - T
        pad_tensor = np.zeros((pad_len, V, C), dtype=float)
        pad_conf = np.zeros((pad_len, V), dtype=float)
        pad_vis = np.zeros((pad_len, V), dtype=float)

        if padding_mode == "repeat" and T > 0:
            last_t = tensor[-1]
            for i in range(pad_len):
                pad_tensor[i] = last_t
                pad_conf[i] = conf[-1]
                pad_vis[i] = vis[-1]

        adj_tensor = np.concatenate([tensor, pad_tensor], axis=0)
        adj_conf = np.concatenate([conf, pad_conf], axis=0)
        adj_vis = np.concatenate([vis, pad_vis], axis=0)

        return adj_tensor, adj_conf, adj_vis

    def clear(self) -> None:
        """Clear internal sequence storage and buffers."""
        self._sequences.clear()
        self._pose_buffers.clear()

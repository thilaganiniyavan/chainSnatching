"""Pose Post-Processor — post-processing for PoseResult sequences.

Implements:
- Linear interpolation for missing/low-confidence keypoints across time series
- EMA temporal coordinate smoothing across consecutive pose samples
- Confidence threshold filtering (zeroing low-confidence joint coordinates)
- Keypoint coordinate normalization relative to bounding box and frame bounds
- Overall pose quality scoring (mean confidence * completeness ratio)
"""

from __future__ import annotations

import numpy as np

from src.core.models.pose_result import PoseResult


class PosePostProcessor:
    """Post-processes time-series PoseResult sequences.

    Args:
        min_joint_confidence: Confidence threshold below which a joint is treated as missing.
        ema_alpha: Exponential Moving Average alpha parameter for temporal smoothing (0.0, 1.0].
        max_missing_gap: Maximum consecutive missing frame gap for linear keypoint interpolation.
    """

    def __init__(
        self,
        min_joint_confidence: float = 0.30,
        ema_alpha: float = 0.45,
        max_missing_gap: int = 10,
    ) -> None:
        self.min_joint_confidence = min_joint_confidence
        self.ema_alpha = min(max(ema_alpha, 0.01), 1.0)
        self.max_missing_gap = max_missing_gap

    def process_sequence(self, poses: list[PoseResult]) -> list[PoseResult]:
        """Post-process a chronological sequence of PoseResult objects for a single person track.

        Args:
            poses: Chronological list of PoseResult instances.

        Returns:
            Post-processed list of PoseResult instances.
        """
        if not poses:
            return []

        num_joints = poses[0].num_keypoints
        n_frames = len(poses)

        # Build 3D array of pixel coordinates & confidences: shape [n_frames, num_joints, 4]
        # (x, y, conf, vis)
        raw_matrix = np.zeros((n_frames, num_joints, 4), dtype=float)
        for i, p in enumerate(poses):
            for j, kp in enumerate(p.keypoints_pixel[:num_joints]):
                raw_matrix[i, j] = kp

        # 1. Linear interpolation for missing joints per keypoint index
        interp_matrix = self._interpolate_keypoint_gaps(raw_matrix)

        # 2. EMA Temporal smoothing per keypoint index
        smooth_matrix = self._smooth_keypoint_coordinates(interp_matrix)

        # 3. Confidence threshold filtering & quality score updating
        result_poses: list[PoseResult] = []

        for i, p in enumerate(poses):
            updated_px: list[tuple[float, float, float, float]] = []
            updated_norm: list[tuple[float, float, float, float]] = []

            x1, y1, x2, y2 = p.bbox_reference
            w_box = max(1.0, float(x2 - x1))
            h_box = max(1.0, float(y2 - y1))

            confs: list[float] = []

            for j in range(num_joints):
                x, y, conf, vis = smooth_matrix[i, j]

                # Filter low-confidence joints
                if conf < self.min_joint_confidence:
                    conf = 0.0
                    vis = 0.0

                confs.append(conf)

                # Re-calculate normalized coordinates relative to bounding box
                x_norm = max(0.0, min(1.0, (x - x1) / w_box))
                y_norm = max(0.0, min(1.0, (y - y1) / h_box))

                updated_px.append((round(float(x), 1), round(float(y), 1), round(float(conf), 4), round(float(vis), 4)))
                updated_norm.append((round(float(x_norm), 4), round(float(y_norm), 4), round(float(conf), 4), round(float(vis), 4)))

            mean_conf = float(np.mean(confs)) if confs else 0.0
            valid_joints = sum(1 for c in confs if c >= self.min_joint_confidence)
            completeness = valid_joints / max(1, num_joints)
            quality_score = round(mean_conf * completeness, 4)

            p.keypoints_pixel = updated_px
            p.keypoints_normalized = updated_norm
            p.overall_confidence = round(mean_conf, 4)
            p.quality_score = quality_score

            result_poses.append(p)

        return result_poses

    def _interpolate_keypoint_gaps(self, matrix: np.ndarray) -> np.ndarray:
        """Interpolate missing keypoint coordinates per joint index across time."""
        n_frames, num_joints, _ = matrix.shape
        result = matrix.copy()

        for j in range(num_joints):
            valid_indices = [
                i for i in range(n_frames)
                if matrix[i, j, 2] >= self.min_joint_confidence
            ]

            if not valid_indices:
                continue

            first_v = valid_indices[0]
            for i in range(first_v):
                result[i, j] = matrix[first_v, j]

            last_v = valid_indices[-1]
            for i in range(last_v + 1, n_frames):
                result[i, j] = matrix[last_v, j]

            for k in range(len(valid_indices) - 1):
                idx1 = valid_indices[k]
                idx2 = valid_indices[k + 1]

                gap_len = idx2 - idx1 - 1
                if 0 < gap_len <= self.max_missing_gap:
                    k1 = matrix[idx1, j]
                    k2 = matrix[idx2, j]
                    for step in range(1, gap_len + 1):
                        t = step / (gap_len + 1)
                        result[idx1 + step, j] = k1 + t * (k2 - k1)

        return result

    def _smooth_keypoint_coordinates(self, matrix: np.ndarray) -> np.ndarray:
        """Apply EMA temporal smoothing to (x, y) joint positions."""
        n_frames, num_joints, _ = matrix.shape
        smoothed = matrix.copy()

        for i in range(1, n_frames):
            target_xy = matrix[i, :, :2]
            prev_xy = smoothed[i - 1, :, :2]
            smoothed[i, :, :2] = self.ema_alpha * target_xy + (1.0 - self.ema_alpha) * prev_xy

        return smoothed

"""Sequence Quality Evaluator — quantitative quality evaluation for SkeletonSequences.

Evaluates:
- Sequence completeness score
- Average keypoint confidence score
- Joint stability score (inverse jitter variance)
- Missing joint ratio
- Sequence continuity score

Applies threshold acceptance checks to set the ``is_accepted`` flag
and ``rejection_reason`` on SkeletonSequence.
"""

from __future__ import annotations

import numpy as np

from src.core.models.skeleton_sequence import SkeletonSequence


class SequenceQualityEvaluator:
    """Evaluates sequence quality and applies threshold acceptance rules.

    Args:
        min_completeness: Minimum required sequence frame completeness ratio [0.0, 1.0].
        min_confidence: Minimum required average keypoint confidence [0.0, 1.0].
        min_stability: Minimum required joint stability score [0.0, 1.0].
        max_missing_ratio: Maximum allowed missing keypoint ratio [0.0, 1.0].
    """

    def __init__(
        self,
        min_completeness: float = 0.50,
        min_confidence: float = 0.40,
        min_stability: float = 0.30,
        max_missing_ratio: float = 0.50,
    ) -> None:
        self.min_completeness = min_completeness
        self.min_confidence = min_confidence
        self.min_stability = min_stability
        self.max_missing_ratio = max_missing_ratio

    def evaluate(self, sequence: SkeletonSequence) -> SkeletonSequence:
        """Compute metrics and set ``is_accepted`` and ``rejection_reason`` on *sequence*.

        Args:
            sequence: The SkeletonSequence object.

        Returns:
            The updated :class:`SkeletonSequence` object.
        """
        tensor = sequence.skeleton_tensor # shape (T, V, 4)

        if tensor.size == 0 or sequence.frame_count == 0:
            sequence.quality_score = 0.0
            sequence.completeness_score = 0.0
            sequence.is_accepted = False
            sequence.rejection_reason = "Empty sequence tensor"
            return sequence

        T, V, C = tensor.shape
        conf_matrix = sequence.joint_confidence_matrix # shape (T, V)

        # 1. Average keypoint confidence
        avg_conf = float(np.mean(conf_matrix)) if conf_matrix.size > 0 else 0.0

        # 2. Missing joint ratio (conf < 0.3)
        missing_count = int(np.sum(conf_matrix < 0.3))
        total_joints = T * V
        missing_ratio = missing_count / max(1, total_joints)

        # 3. Completeness score
        valid_frames_count = sum(
            1 for t in range(T) if np.mean(conf_matrix[t]) >= 0.25
        )
        completeness = valid_frames_count / max(1, T)

        # 4. Joint stability (1.0 / (1.0 + mean_displacement_std))
        if T > 1:
            diffs = []
            for t in range(1, T):
                p1 = tensor[t - 1, :, :2]
                p2 = tensor[t, :, :2]
                disp = np.linalg.norm(p2 - p1, axis=1)
                diffs.append(np.mean(disp))
            std_jitter = float(np.std(diffs)) if diffs else 0.0
            stability = 1.0 / (1.0 + std_jitter)
        else:
            stability = 1.0

        overall_quality = round(avg_conf * (1.0 - missing_ratio) * completeness * stability, 4)

        sequence.quality_score = overall_quality
        sequence.completeness_score = round(float(completeness), 4)

        # Threshold acceptance rules
        rejections: list[str] = []

        if completeness < self.min_completeness:
            rejections.append(
                f"Completeness {completeness:.2f} < threshold {self.min_completeness:.2f}"
            )
        if avg_conf < self.min_confidence:
            rejections.append(
                f"Confidence {avg_conf:.2f} < threshold {self.min_confidence:.2f}"
            )
        if stability < self.min_stability:
            rejections.append(
                f"Stability {stability:.2f} < threshold {self.min_stability:.2f}"
            )
        if missing_ratio > self.max_missing_ratio:
            rejections.append(
                f"Missing ratio {missing_ratio:.2f} > threshold {self.max_missing_ratio:.2f}"
            )

        if rejections:
            sequence.is_accepted = False
            sequence.rejection_reason = "; ".join(rejections)
        else:
            sequence.is_accepted = True
            sequence.rejection_reason = "Accepted"

        return sequence

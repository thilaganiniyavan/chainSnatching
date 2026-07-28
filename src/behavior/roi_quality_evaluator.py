"""ROI Quality Evaluator — quantitative quality evaluation for candidate ROIs.

Computes:
- ROI completeness ratio
- Missing detection percentage
- Bounding-box stability score
- Track continuity score
- Frame coverage score

Accepts or rejects candidate ROIs against configurable quality thresholds,
setting the ``is_accepted`` flag and ``rejection_reason`` on InteractionROI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.models.interaction_roi import InteractionROI


class ROIQualityEvaluator:
    """Evaluates quantitative quality metrics and applies acceptance thresholds to ROIs.

    Args:
        min_completeness: Minimum acceptable detection completeness ratio [0.0, 1.0].
        max_missing_pct: Maximum allowed missing detection percentage (0.0 to 100.0).
        min_stability: Minimum acceptable bounding box stability score [0.0, 1.0].
        min_coverage: Minimum acceptable frame coverage score [0.0, 1.0].
    """

    def __init__(
        self,
        min_completeness: float = 0.60,
        max_missing_pct: float = 40.0,
        min_stability: float = 0.35,
        min_coverage: float = 0.50,
    ) -> None:
        self.min_completeness = min_completeness
        self.max_missing_pct = max_missing_pct
        self.min_stability = min_stability
        self.min_coverage = min_coverage

    def evaluate(
        self,
        roi: InteractionROI,
        raw_box_mask: list[bool] | None = None,
    ) -> InteractionROI:
        """Compute quality metrics and update ``is_accepted`` and ``rejection_reason`` on *roi*.

        Args:
            roi: The InteractionROI domain model instance.
            raw_box_mask: Optional boolean list indicating which frames had valid raw detections.

        Returns:
            The updated :class:`InteractionROI` instance.
        """
        boxes = roi.bounding_box_sequence
        total_frames = max(1, roi.frame_count)

        if not boxes:
            roi.quality_metrics = {
                "completeness": 0.0,
                "missing_detection_percentage": 100.0,
                "bounding_box_stability": 0.0,
                "track_continuity": 0.0,
                "frame_coverage": 0.0,
            }
            roi.is_accepted = False
            roi.rejection_reason = "Empty bounding box sequence"
            return roi

        # Compute raw detection mask if not explicitly passed
        if raw_box_mask is None:
            raw_box_mask = [b != (0, 0, 0, 0) and b != (0, 0, 100, 100) for b in boxes]

        valid_count = sum(1 for m in raw_box_mask if m)
        missing_count = total_frames - valid_count

        # 1. Completeness ratio [0.0, 1.0]
        completeness = valid_count / total_frames

        # 2. Missing detection percentage [0.0, 100.0]
        missing_pct = (missing_count / total_frames) * 100.0

        # 3. Track continuity (max consecutive missing gap)
        max_gap = 0
        current_gap = 0
        for is_valid in raw_box_mask:
            if not is_valid:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        continuity = max(0.0, 1.0 - (max_gap / total_frames))

        # 4. Bounding box stability (1.0 / (1.0 + std_area_variance))
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        if len(areas) > 1 and np.mean(areas) > 0:
            norm_areas = np.array(areas, dtype=float) / np.mean(areas)
            std_var = float(np.std(norm_areas))
            stability = 1.0 / (1.0 + std_var)
        else:
            stability = 1.0

        # 5. Frame coverage
        coverage = min(1.0, total_frames / max(1, (roi.end_frame - roi.start_frame + 1)))

        metrics = {
            "completeness": round(float(completeness), 4),
            "missing_detection_percentage": round(float(missing_pct), 2),
            "bounding_box_stability": round(float(stability), 4),
            "track_continuity": round(float(continuity), 4),
            "frame_coverage": round(float(coverage), 4),
        }
        roi.quality_metrics = metrics

        # Apply threshold acceptance checks
        rejection_reasons: list[str] = []

        if completeness < self.min_completeness:
            rejection_reasons.append(
                f"Completeness {completeness:.2f} < threshold {self.min_completeness:.2f}"
            )
        if missing_pct > self.max_missing_pct:
            rejection_reasons.append(
                f"Missing pct {missing_pct:.1f}% > threshold {self.max_missing_pct:.1f}%"
            )
        if stability < self.min_stability:
            rejection_reasons.append(
                f"Stability {stability:.2f} < threshold {self.min_stability:.2f}"
            )
        if coverage < self.min_coverage:
            rejection_reasons.append(
                f"Coverage {coverage:.2f} < threshold {self.min_coverage:.2f}"
            )

        if rejection_reasons:
            roi.is_accepted = False
            roi.rejection_reason = "; ".join(rejection_reasons)
        else:
            roi.is_accepted = True
            roi.rejection_reason = "Accepted"

        return roi

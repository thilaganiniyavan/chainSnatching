"""OpenPose Framework Adapter Scaffold.

Allows future integration of OpenPose framework models without changing downstream modules.
"""

from __future__ import annotations

import numpy as np

from src.core.models.pose_result import PoseResult
from src.core.models.interaction_roi import PreparedSkeletonSample
from src.pose.base_estimator import AbstractPoseEstimator
from src.pose.mediapipe_estimator import MediaPipePoseEstimator


class OpenPoseAdapter(AbstractPoseEstimator):
    """Adapter scaffold for OpenPose framework."""

    def __init__(self, model_config: str = "body_25") -> None:
        super().__init__(backend_name="OpenPose")
        self.model_config = model_config
        self._fallback = MediaPipePoseEstimator(topology="COCO_17")

    def estimate_pose(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_index: int,
        timestamp: float,
        track_id: int,
        interaction_id: str = "",
        roi_id: str = "",
    ) -> PoseResult:
        res = self._fallback.estimate_pose(
            image, bbox, frame_index, timestamp, track_id, interaction_id, roi_id
        )
        res.backend_name = "OpenPose"
        res.metadata["scaffold_adapter"] = True
        return res

    def estimate_batch(
        self,
        samples: list[PreparedSkeletonSample],
        frames_dict: dict[int, np.ndarray],
    ) -> list[PoseResult]:
        results = self._fallback.estimate_batch(samples, frames_dict)
        for r in results:
            r.backend_name = "OpenPose"
            r.metadata["scaffold_adapter"] = True
        return results

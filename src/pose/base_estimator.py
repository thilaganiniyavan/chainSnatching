"""Abstract Pose Estimator Interface.

Defines the contract for all pose estimation backends and framework adapters.
Every implementation must produce standardized PoseResult objects given a
cropped person image ROI and tracking metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.core.models.pose_result import PoseResult
from src.core.models.interaction_roi import PreparedSkeletonSample


class AbstractPoseEstimator(ABC):
    """Abstract base class for all pose estimation models and framework adapters."""

    def __init__(self, backend_name: str = "Abstract") -> None:
        self.backend_name = backend_name

    @abstractmethod
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
        """Estimate keypoints for a single person crop image.

        Args:
            image: Full video frame or cropped person image ROI (BGR, uint8).
            bbox: Person bounding box ``(x1, y1, x2, y2)``.
            frame_index: Frame number.
            timestamp: Timestamp in seconds.
            track_id: Person track ID.
            interaction_id: Source interaction ID.
            roi_id: Source InteractionROI ID.

        Returns:
            A :class:`PoseResult` object.
        """
        pass

    @abstractmethod
    def estimate_batch(
        self,
        samples: list[PreparedSkeletonSample],
        frames_dict: dict[int, np.ndarray],
    ) -> list[PoseResult]:
        """Estimate poses for a batch of PreparedSkeletonSample instances.

        Args:
            samples: List of PreparedSkeletonSample objects.
            frames_dict: Mapping frame_number -> BGR numpy frame.

        Returns:
            List of :class:`PoseResult` objects.
        """
        pass

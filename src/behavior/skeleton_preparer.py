"""Skeleton Preparer — standardized input sample generator for Pose Estimation.

Does NOT execute pose estimation directly.  Instead, prepares framework-agnostic
PreparedSkeletonSample objects containing cropped ROI coordinates, frame timestamps,
and standardized skeleton schema placeholders compatible with MediaPipe, RTMPose,
ViTPose, MMPose, and OpenPose models.
"""

from __future__ import annotations

from typing import Any

from src.core.models.interaction_roi import InteractionROI, PreparedSkeletonSample


class SkeletonPreparer:
    """Generates framework-agnostic pose input samples from accepted InteractionROIs."""

    def __init__(self) -> None:
        pass

    def prepare_samples(
        self,
        roi: InteractionROI,
    ) -> list[PreparedSkeletonSample]:
        """Generate a list of PreparedSkeletonSample instances for each frame in *roi*.

        Args:
            roi: An accepted :class:`InteractionROI` instance.

        Returns:
            List of :class:`PreparedSkeletonSample` instances.
        """
        if not roi.is_accepted or not roi.frame_index_mapping:
            return []

        samples: list[PreparedSkeletonSample] = []

        for idx, frame_num in enumerate(roi.frame_index_mapping):
            raw_box = roi.bounding_box_sequence[idx] if idx < len(roi.bounding_box_sequence) else (0, 0, 0, 0)
            exp_box = roi.expanded_bounding_boxes[idx] if idx < len(roi.expanded_bounding_boxes) else raw_box
            timestamp = roi.timestamps[idx] if idx < len(roi.timestamps) else 0.0

            placeholder_schema = {
                "topology": "COCO_17",
                "num_keypoints": 17,
                "keypoint_names": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ],
                "keypoints_2d": None,      # Placeholder for shape [17, 3] (x, y, confidence)
                "pose_score": None,
                "model_compatibility": ["MediaPipe", "RTMPose", "ViTPose", "MMPose", "OpenPose"],
            }

            sample = PreparedSkeletonSample(
                roi_id=roi.roi_id,
                interaction_id=roi.interaction_id,
                frame_number=frame_num,
                timestamp=timestamp,
                person_track_id=roi.person_track_id,
                raw_bbox=raw_box,
                expanded_bbox=exp_box,
                expected_skeleton_placeholder=placeholder_schema,
                metadata={
                    "sequence_index": idx,
                    "total_sequence_frames": roi.frame_count,
                    "interaction_confidence": roi.interaction_confidence,
                    "graph_reference_id": roi.graph_reference_id,
                },
            )

            samples.append(sample)

        return samples

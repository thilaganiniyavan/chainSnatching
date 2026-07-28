"""Pipeline Stage for the Pose Estimation Abstraction Layer.

Consumes accepted Interaction ROIs from upstream stage, runs model-agnostic pose
estimation via :class:`PoseEstimatorFactory`, post-processes keypoints via
:class:`PosePostProcessor`, annotates frames via :class:`PoseOverlayVisualizer`,
and logs datasets via :class:`PoseLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.pose.factory import PoseEstimatorFactory
from src.pose.pose_post_processor import PosePostProcessor
from src.pose.pose_visualizer import PoseOverlayVisualizer, PosePreviewExporter
from src.pose.pose_logger import PoseLogger


class PoseEstimationStage(Stage):
    """Pipeline stage executing model-agnostic pose estimation on selected Interaction ROIs.

    Args:
        backend_name: Name of pose backend ("mediapipe", "rtmpose", "vitpose", "mmpose", "openpose").
        fps: Video frame rate.
        min_joint_confidence: Confidence threshold for keypoint post-processing.
        output_json_path: Path for pose_results.json export.
        output_csv_path: Path for pose_statistics.csv export.
        output_report_path: Path for pose_quality_report.md export.
        export_previews_dir: Optional directory path to export rendered pose preview videos.
    """

    def __init__(
        self,
        backend_name: str = "mediapipe",
        fps: float = 30.0,
        min_joint_confidence: float = 0.30,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_previews_dir: str | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.estimator = PoseEstimatorFactory.create(backend_name=backend_name)
        self.post_processor = PosePostProcessor(min_joint_confidence=min_joint_confidence)
        self.visualizer = PoseOverlayVisualizer()
        self.logger = PoseLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_previews_dir = export_previews_dir

        self._evaluated_sample_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, run pose estimation on active accepted ROIs."""

        roi_engine = context.metadata.get("roi_engine")
        accepted_rois = context.metadata.get("accepted_rois", [])

        current_frame_poses = []

        if roi_engine and accepted_rois:
            for roi in accepted_rois:
                if context.frame_number in roi.frame_index_mapping:
                    samples = roi_engine.prepare_skeleton_samples(roi)
                    # Filter samples for the current frame
                    current_samples = [s for s in samples if s.frame_number == context.frame_number]

                    for s in current_samples:
                        pose_res = self.estimator.estimate_pose(
                            image=context.frame,
                            bbox=s.expanded_bbox,
                            frame_index=s.frame_number,
                            timestamp=s.timestamp,
                            track_id=s.person_track_id,
                            interaction_id=s.interaction_id,
                            roi_id=s.roi_id,
                        )
                        current_frame_poses.append(pose_res)
                        if s.sample_id not in self._evaluated_sample_ids:
                            self.logger.log_pose(pose_res)
                            self._evaluated_sample_ids.add(s.sample_id)

        context.poses = current_frame_poses
        context.metadata["pose_results"] = current_frame_poses
        context.metadata["pose_estimator"] = self.estimator

        # Render skeleton overlays on output video frame
        base_frame = context.metadata.get(
            "roi_frame",
            context.metadata.get(
                "graph_frame",
                context.metadata.get(
                    "reasoning_frame",
                    context.metadata.get(
                        "behaviour_frame",
                        context.metadata.get(
                            "relationship_frame",
                            context.metadata.get("trajectory_frame", context.frame),
                        ),
                    ),
                ),
            ),
        )

        pose_frame = self.visualizer.draw(base_frame, current_frame_poses)
        context.metadata["pose_frame"] = pose_frame

        return context

    def finalize(self) -> None:
        """Export accumulated pose results to JSON, CSV, and quality report files."""
        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

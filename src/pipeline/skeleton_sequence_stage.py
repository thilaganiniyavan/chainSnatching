"""Pipeline Stage for the Skeleton Sequence Builder.

Consumes PoseResult objects from upstream PoseEstimationStage, executes
:class:`SkeletonSequenceBuilder` to build spatially normalized and validated
:class:`SkeletonSequence` objects, annotates frames via
:class:`SkeletonSequenceVisualizer`, and logs datasets via :class:`SkeletonSequenceLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.skeleton_sequence_builder import SkeletonSequenceBuilder
from src.behavior.skeleton_normalizer import SkeletonNormalizer
from src.behavior.sequence_quality_evaluator import SequenceQualityEvaluator
from src.behavior.skeleton_sequence_visualizer import SkeletonSequenceVisualizer, SequencePreviewExporter
from src.behavior.skeleton_sequence_logger import SkeletonSequenceLogger


class SkeletonSequenceStage(Stage):
    """Pipeline stage building normalized SkeletonSequence objects for downstream action recognition models.

    Args:
        normalization_method: Normalization strategy ("hip_centered", "bbox", "root_joint", "image").
        fps: Video frame rate.
        output_json_path: Path for skeleton_sequences.json export.
        output_csv_path: Path for sequence_statistics.csv export.
        output_report_path: Path for sequence_quality_report.md export.
        export_previews_dir: Optional directory path to export preview videos.
    """

    def __init__(
        self,
        normalization_method: str = "hip_centered",
        fps: float = 30.0,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_previews_dir: str | None = None,
    ) -> None:
        normalizer = SkeletonNormalizer(method=normalization_method)
        evaluator = SequenceQualityEvaluator()

        self.builder = SkeletonSequenceBuilder(
            normalizer=normalizer,
            evaluator=evaluator,
            fps=fps,
        )
        self.visualizer = SkeletonSequenceVisualizer()
        self.logger = SkeletonSequenceLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_previews_dir = export_previews_dir

        self._logged_sequence_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, append PoseResults, and finalize active sequences."""

        poses = context.poses

        # 1. Append frame PoseResults to active sequence buffers
        for p in poses:
            seq_id = f"SEQ-{p.interaction_id}-TRK-{p.track_id}"
            self.builder.append_pose(seq_id, p)

        # Finalize active sequences for completed interactions
        accepted_rois = context.metadata.get("accepted_rois", [])
        for roi in accepted_rois:
            seq_id = f"SEQ-{roi.interaction_id}-TRK-{roi.person_track_id}"
            if roi.end_frame <= context.frame_number:
                self.builder.finalize_sequence(seq_id)

        all_seqs = self.builder.get_completed_sequences()
        accepted_seqs = [s for s in all_seqs if s.is_accepted]

        context.sequences = accepted_seqs
        context.metadata["skeleton_sequences"] = all_seqs
        context.metadata["accepted_sequences"] = accepted_seqs
        context.metadata["sequence_builder"] = self.builder

        # 2. Render sequence HUD overlay on video frame
        base_frame = context.metadata.get(
            "pose_frame",
            context.metadata.get(
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
            ),
        )

        sequence_frame = self.visualizer.draw(base_frame, all_seqs, context.frame_number)
        context.metadata["sequence_frame"] = sequence_frame

        # 3. Log completed sequences
        for s in all_seqs:
            if s.sequence_id not in self._logged_sequence_ids:
                self.logger.log_sequence(s)
                self._logged_sequence_ids.add(s.sequence_id)

        return context

    def finalize(self) -> None:
        """Export accumulated SkeletonSequences to JSON, CSV, and quality report files."""
        for s in self.builder.get_completed_sequences():
            if s.sequence_id not in self._logged_sequence_ids:
                self.logger.log_sequence(s)
                self._logged_sequence_ids.add(s.sequence_id)

        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

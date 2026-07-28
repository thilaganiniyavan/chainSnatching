"""Pipeline Stage for Human Action Recognition.

Consumes SkeletonSequence objects from upstream SkeletonSequenceStage, executes
model-agnostic action recognition via :class:`ActionRecognizerFactory`, post-processes
predictions via :class:`ActionPostProcessor`, annotates video frames via
:class:`ActionOverlayVisualizer`, and logs datasets via :class:`ActionLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.action.factory import ActionRecognizerFactory
from src.action.action_post_processor import ActionPostProcessor
from src.action.action_visualizer import ActionOverlayVisualizer, ActionPreviewExporter
from src.action.action_logger import ActionLogger


class ActionRecognitionStage(Stage):
    """Pipeline stage classifying human physical actions from SkeletonSequence tensors.

    Args:
        backend_name: Name of action recognizer model backend ("stgcn", "ctrgcn", "msg3d", "posec3d").
        min_confidence: Confidence threshold for action prediction fallback.
        fps: Video frame rate.
        output_json_path: Path for action_results.json export.
        output_csv_path: Path for action_statistics.csv export.
        output_report_path: Path for action_recognition_report.md export.
        export_previews_dir: Optional directory path to export preview videos.
    """

    def __init__(
        self,
        backend_name: str = "stgcn",
        min_confidence: float = 0.40,
        fps: float = 30.0,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_previews_dir: str | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.recognizer = ActionRecognizerFactory.create(backend_name=backend_name)
        self.post_processor = ActionPostProcessor(min_confidence=min_confidence)
        self.visualizer = ActionOverlayVisualizer()
        self.logger = ActionLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_previews_dir = export_previews_dir

        self._classified_sequence_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, classify actions for active accepted SkeletonSequences."""

        accepted_sequences = context.metadata.get("accepted_sequences", [])
        current_frame_actions = []

        for seq in accepted_sequences:
            if context.frame_number in seq.frame_indices:
                raw_res = self.recognizer.predict_action(seq)
                processed_res = self.post_processor.process(raw_res)

                current_frame_actions.append(processed_res)

                if seq.sequence_id not in self._classified_sequence_ids:
                    self.logger.log_result(processed_res)
                    self._classified_sequence_ids.add(seq.sequence_id)

        context.actions = current_frame_actions
        context.metadata["action_results"] = current_frame_actions
        context.metadata["action_recognizer"] = self.recognizer

        # Render action HUD overlay on video frame
        base_frame = context.metadata.get(
            "sequence_frame",
            context.metadata.get(
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
            ),
        )

        action_frame = self.visualizer.draw(base_frame, current_frame_actions)
        context.metadata["action_frame"] = action_frame

        return context

    def finalize(self) -> None:
        """Export accumulated ActionResults to JSON, CSV, and quality report files."""
        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

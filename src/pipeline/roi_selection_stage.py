"""Pipeline Stage for Interaction ROI Selection & Skeleton Preparation.

Consumes Behaviour Graphs and Track histories, executes :class:`ROIEngine` to
select, process, and evaluate Interaction ROIs, renders video overlays via
:class:`ROIOverlayVisualizer`, and logs output via :class:`ROILogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.roi_engine import ROIEngine
from src.behavior.roi_bbox_processor import ROIBBoxProcessor
from src.behavior.roi_quality_evaluator import ROIQualityEvaluator
from src.behavior.roi_visualizer import ROIOverlayVisualizer, ROIClipExporter
from src.behavior.roi_logger import ROILogger


class ROISelectionStage(Stage):
    """Pipeline stage selecting and evaluating Interaction ROIs for downstream pose estimation.

    Args:
        fps: Frame rate of input video.
        bbox_processor: Custom ROIBBoxProcessor instance.
        quality_evaluator: Custom ROIQualityEvaluator instance.
        output_json_path: Path for interaction_rois.json export.
        output_csv_path: Path for roi_statistics.csv export.
        output_report_path: Path for roi_quality_report.md export.
        export_clips_dir: Optional directory path to export cropped video clips (.mp4/.avi).
    """

    def __init__(
        self,
        fps: float = 30.0,
        bbox_processor: ROIBBoxProcessor | None = None,
        quality_evaluator: ROIQualityEvaluator | None = None,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_clips_dir: str | None = None,
    ) -> None:
        self.engine = ROIEngine(
            bbox_processor=bbox_processor,
            quality_evaluator=quality_evaluator,
            fps=fps,
        )
        self.visualizer = ROIOverlayVisualizer(fps=fps)
        self.logger = ROILogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_clips_dir = export_clips_dir

        self._logged_roi_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, update ROIs, and render video overlay."""

        graphs = context.metadata.get("behaviour_graphs", [])
        tracks = context.tracks

        # 1. Update ROIs for all active graphs
        active_rois = []
        for graph in graphs:
            roi = self.engine.update_roi(graph, tracks, context.frame_number)
            active_rois.append(roi)

        all_rois = self.engine.get_completed_rois()
        accepted_rois = self.engine.get_accepted_rois()

        context.rois = accepted_rois
        context.metadata["rois"] = all_rois
        context.metadata["accepted_rois"] = accepted_rois
        context.metadata["roi_engine"] = self.engine

        # 2. Render HUD overlay on video frame
        base_frame = context.metadata.get(
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
        )

        roi_frame = self.visualizer.draw(base_frame, all_rois, context.frame_number)
        context.metadata["roi_frame"] = roi_frame

        # 3. Log completed ROIs
        for r in all_rois:
            if r.roi_id not in self._logged_roi_ids:
                self.logger.log_roi(r)
                self._logged_roi_ids.add(r.roi_id)

        return context

    def finalize(self) -> None:
        """Export accumulated ROIs to JSON, CSV, and quality report files."""
        for r in self.engine.get_completed_rois():
            if r.roi_id not in self._logged_roi_ids:
                self.logger.log_roi(r)
                self._logged_roi_ids.add(r.roi_id)

        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

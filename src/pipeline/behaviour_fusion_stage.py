"""Pipeline Stage for the Behaviour Fusion Engine.

Consumes Behaviour Graphs (Stream A) and Action Results (Stream B), executes
:class:`BehaviourFusionEngine` to fuse evidence into multi-modal :class:`FusedInteraction`
objects, annotates video frames via :class:`FusionOverlayVisualizer`, and logs datasets
via :class:`FusionLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.behaviour_fusion_engine import BehaviourFusionEngine
from src.behavior.fusion_strategies import FusionStrategyEngine
from src.behavior.fusion_explainer import FusionExplainer
from src.behavior.fusion_visualizer import FusionOverlayVisualizer, FusionPreviewExporter
from src.behavior.fusion_logger import FusionLogger


class BehaviourFusionStage(Stage):
    """Pipeline stage executing multi-modal evidence fusion across Behaviour Graphs and Action Recognition.

    Args:
        fusion_strategy: Name of fusion strategy ("weighted_confidence", "bayesian", "rule_based", "voting_based", "weighted_averaging").
        fps: Video frame rate.
        output_json_path: Path for fused_interactions.json export.
        output_csv_path: Path for fusion_statistics.csv export.
        output_report_path: Path for fusion_report.md export.
        export_previews_dir: Optional directory path to export preview videos.
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted_confidence",
        fps: float = 30.0,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
        output_report_path: str | None = None,
        export_previews_dir: str | None = None,
    ) -> None:
        self.engine = BehaviourFusionEngine(
            fusion_strategy=fusion_strategy,
            fps=fps,
        )
        self.visualizer = FusionOverlayVisualizer()
        self.logger = FusionLogger()

        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.output_report_path = output_report_path
        self.export_previews_dir = export_previews_dir

        self._fused_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, fuse active graphs with action results."""

        graphs = context.metadata.get("behaviour_graphs", [])
        action_results = context.metadata.get("action_results", [])
        tracks = context.tracks

        current_frame_fusions = []

        for graph in graphs:
            fused = self.engine.fuse_interaction(graph, action_results, tracks=tracks)
            current_frame_fusions.append(fused)

            if fused.fusion_id not in self._fused_ids:
                self.logger.log_fusion(fused)
                self._fused_ids.add(fused.fusion_id)

        all_fusions = self.engine.get_completed_fusions()

        context.fused_interactions = all_fusions
        context.metadata["fused_interactions"] = all_fusions
        context.metadata["fusion_engine"] = self.engine

        # Render fusion HUD overlay on video frame
        base_frame = context.metadata.get(
            "action_frame",
            context.metadata.get(
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
            ),
        )

        fusion_frame = self.visualizer.draw(base_frame, all_fusions, context.frame_number)
        context.metadata["fusion_frame"] = fusion_frame

        return context

    def finalize(self) -> None:
        """Export accumulated FusedInteractions to JSON, CSV, and report files."""
        for f in self.engine.get_completed_fusions():
            if f.fusion_id not in self._fused_ids:
                self.logger.log_fusion(f)
                self._fused_ids.add(f.fusion_id)

        if self.output_json_path and self.output_csv_path and self.output_report_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_csv_path,
                self.output_report_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

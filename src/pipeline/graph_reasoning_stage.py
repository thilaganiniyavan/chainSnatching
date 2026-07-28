"""Pipeline Stage for the Behaviour Graph Reasoning Engine.

Consumes Interaction objects, primitives, and timelines from upstream stages,
updates directed Behaviour Graphs via :class:`BehaviourGraphEngine`, annotates
frames via :class:`OverlayVisualizer`, and logs output via :class:`GraphLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.behaviour_graph_engine import BehaviourGraphEngine
from src.behavior.graph_visualizer import OverlayVisualizer, GraphDiagramExporter
from src.behavior.graph_logger import GraphLogger
from src.behavior.pattern_rules import PatternConfig


class GraphReasoningStage(Stage):
    """Pipeline stage building directed Behaviour Graphs across video frames.

    Args:
        fps: Frame rate of input video.
        pattern_config: Optional PatternConfig for PatternEvaluator tuning.
        output_json_path: Path for behaviour_graph.json export.
        output_patterns_csv_path: Path for behaviour_patterns.csv export.
        output_transition_csv_path: Path for transition_matrix.csv export.
        export_diagrams_dir: Optional directory path to export rendered graph diagrams (.png).
    """

    def __init__(
        self,
        fps: float = 30.0,
        pattern_config: PatternConfig | None = None,
        output_json_path: str | None = None,
        output_patterns_csv_path: str | None = None,
        output_transition_csv_path: str | None = None,
        export_diagrams_dir: str | None = None,
    ) -> None:
        self.engine = BehaviourGraphEngine(pattern_config=pattern_config, fps=fps)
        self.visualizer = OverlayVisualizer(fps=fps)
        self.logger = GraphLogger()

        self.output_json_path = output_json_path
        self.output_patterns_csv_path = output_patterns_csv_path
        self.output_transition_csv_path = output_transition_csv_path
        self.export_diagrams_dir = export_diagrams_dir

        self._logged_graph_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context, update graphs, and render overlays."""

        interactions = context.metadata.get("interactions", [])
        primitives = context.metadata.get("behaviours", [])
        timeline_mgr = context.metadata.get("behaviour_timeline")

        timelines = timeline_mgr.get_all_timelines() if timeline_mgr else {}

        # 1. Update directed graph for each interaction
        active_graphs = []
        for interaction in interactions:
            tl = timelines.get(interaction.interaction_id, [])
            graph = self.engine.update_graph(
                interaction, primitives, tl, context.frame_number
            )
            active_graphs.append(graph)

        all_graphs = self.engine.get_all_graphs()
        context.graphs = all_graphs
        context.metadata["behaviour_graphs"] = all_graphs
        context.metadata["graph_engine"] = self.engine

        # 2. Render HUD overlay on video frame
        base_frame = context.metadata.get(
            "reasoning_frame",
            context.metadata.get(
                "behaviour_frame",
                context.metadata.get(
                    "relationship_frame",
                    context.metadata.get("trajectory_frame", context.frame),
                ),
            ),
        )

        graph_frame = self.visualizer.draw(base_frame, all_graphs, context.tracks)
        context.metadata["graph_frame"] = graph_frame

        # 3. Log completed graphs
        completed = self.engine.get_completed_graphs()
        for g in completed:
            if g.graph_id not in self._logged_graph_ids:
                self.logger.log_graph(g)
                self._logged_graph_ids.add(g.graph_id)

        return context

    def finalize(self) -> None:
        """Export accumulated graphs to JSON, CSVs, and graph diagrams."""
        # Include active graphs on finalize if not logged
        for g in self.engine.get_all_graphs():
            if g.graph_id not in self._logged_graph_ids:
                self.logger.log_graph(g)
                self._logged_graph_ids.add(g.graph_id)

        all_logged = self.logger.get_graphs()

        if self.output_json_path and self.output_patterns_csv_path and self.output_transition_csv_path:
            self.logger.export_all(
                self.output_json_path,
                self.output_patterns_csv_path,
                self.output_transition_csv_path,
            )
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)

        if self.export_diagrams_dir:
            GraphDiagramExporter.export_all_diagrams(all_logged, self.export_diagrams_dir)

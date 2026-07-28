"""Pipeline Stage for the Behaviour Reasoning Engine.

Consumes Interaction objects and Behaviour Timelines from upstream stages,
runs rule-graph classification via :class:`ReasoningEngine`, annotates
frames via :class:`EventVisualizer`, and logs output via :class:`EventLogger`.
"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.reasoning_engine import ReasoningEngine
from src.behavior.event_visualizer import EventVisualizer
from src.behavior.event_logger import EventLogger
from src.behavior.reasoning_rules import RuleNode


class ReasoningStage(Stage):
    """Pipeline stage executing rule-graph reasoning on interactions.

    Args:
        fps: Frame rate of input video.
        rules: Optional custom rule graph.
        output_json_path: Path for behaviour_events.json export.
        output_csv_path: Path for behaviour_events.csv export.
    """

    def __init__(
        self,
        fps: float = 30.0,
        rules: list[RuleNode] | None = None,
        output_json_path: str | None = None,
        output_csv_path: str | None = None,
    ) -> None:
        self.engine = ReasoningEngine(rules=rules, fps=fps)
        self.visualizer = EventVisualizer(fps=fps)
        self.logger = EventLogger()
        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path

        self._evaluated_interaction_ids: set[str] = set()

    def process(self, context: FrameContext) -> FrameContext:
        """Process current frame context and execute reasoning."""

        interactions = context.metadata.get("interactions", [])
        timeline = context.metadata.get("behaviour_timeline")

        if not timeline:
            return context

        timelines = timeline.get_all_timelines()

        # 1. Real-time tentative reasoning on active interactions
        active_events = self.engine.analyse_all(
            interactions, timelines, tentative=True
        )

        # 2. Final reasoning on completed interactions
        completed = context.metadata.get("completed_interactions", [])
        newly_completed_events = []
        for interaction in completed:
            if interaction.interaction_id not in self._evaluated_interaction_ids:
                tl = timelines.get(interaction.interaction_id, [])
                events = self.engine.analyse_interaction(
                    interaction, tl, tentative=False
                )
                newly_completed_events.extend(events)
                self.logger.log_events(events)
                self._evaluated_interaction_ids.add(interaction.interaction_id)

        # Combine active (tentative) and newly completed events
        current_events = active_events + newly_completed_events
        context.metadata["behaviour_events"] = current_events
        context.metadata["completed_behaviour_events"] = self.logger.get_events()

        # 3. Visualize
        base_frame = context.metadata.get(
            "behaviour_frame",
            context.metadata.get(
                "relationship_frame",
                context.metadata.get("trajectory_frame", context.frame),
            ),
        )

        reasoning_frame = self.visualizer.draw(
            base_frame, current_events, interactions, context.tracks
        )
        context.metadata["reasoning_frame"] = reasoning_frame

        return context

    def finalize(self) -> None:
        """Export accumulated events to JSON/CSV files."""
        if self.output_json_path and self.output_csv_path:
            self.logger.export_all(self.output_json_path, self.output_csv_path)
        elif self.output_json_path:
            self.logger.export_json(self.output_json_path)
        elif self.output_csv_path:
            self.logger.export_csv(self.output_csv_path)

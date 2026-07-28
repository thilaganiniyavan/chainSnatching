"""Pipeline stage that orchestrates the Behaviour Intelligence Layer.

Wires together the :class:`BehaviourEngine`, :class:`BehaviourTimeline`,
:class:`BehaviourVisualizer`, and :class:`BehaviourLogger` into a single
pipeline stage.  On each frame it:

1. Extracts behavioural primitives from active interactions.
2. Records timeline events.
3. Renders behaviour visualization on the output frame.
4. Logs completed interactions to structured JSON.
"""

from __future__ import annotations

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.core.models.interaction import InteractionState
from src.behavior.behaviour_engine import BehaviourEngine
from src.behavior.behaviour_timeline import BehaviourTimeline
from src.behavior.behaviour_visualizer import BehaviourVisualizer
from src.behavior.behaviour_logger import BehaviourLogger


class BehaviourStage(Stage):
    """Pipeline stage that runs behavioural analysis on interactions.

    This stage expects ``context.metadata["interactions"]`` to be populated
    by an upstream :class:`InteractionStage`.

    Args:
        fps: Video FPS for timeline timestamps and duration display.
        output_log_path: Path for the behaviour JSON log file.
            If ``None``, logging is deferred until explicit export.
        **engine_kwargs: Forwarded to :class:`BehaviourEngine` constructor
            for threshold configuration.
    """

    def __init__(
        self,
        fps: float = 30.0,
        output_log_path: str | None = None,
        **engine_kwargs,
    ) -> None:
        self.engine = BehaviourEngine(**engine_kwargs)
        self.timeline = BehaviourTimeline(fps=fps)
        self.visualizer = BehaviourVisualizer(fps=fps)
        self.logger = BehaviourLogger(fps=fps)
        self.output_log_path = output_log_path

        # Track which interactions have already been logged
        self._logged_ids: set[str] = set()

        # Chronological behaviour history (all frames)
        self._behaviour_history: list[dict] = []

    def process(self, context: FrameContext) -> FrameContext:
        """Analyse interactions, update timeline, visualize, and log."""

        interactions = context.metadata.get("interactions", [])

        # 1. Extract behavioural primitives
        behaviours = self.engine.analyse(interactions, context.frame_number)
        context.metadata["behaviours"] = behaviours

        # Store in chronological history
        if behaviours:
            self._behaviour_history.append(
                {
                    "frame": context.frame_number,
                    "primitives": [
                        {
                            "type": bp.primitive_type,
                            "interaction_id": bp.interaction_id,
                            "confidence": bp.confidence,
                        }
                        for bp in behaviours
                    ],
                }
            )

        # 2. Record timeline events for each interaction
        for interaction in interactions:
            interaction_behaviours = [
                bp for bp in behaviours
                if bp.interaction_id == interaction.interaction_id
            ]
            self.timeline.record(
                interaction, interaction_behaviours, context.frame_number
            )

        # 3. Visualize
        base_frame = context.metadata.get(
            "relationship_frame",
            context.metadata.get("trajectory_frame", context.frame),
        )
        viz_frame = self.visualizer.draw(
            base_frame, interactions, behaviours, context.tracks
        )
        context.metadata["behaviour_frame"] = viz_frame

        # 4. Log completed interactions
        completed = context.metadata.get("completed_interactions", [])
        for interaction in completed:
            if interaction.interaction_id not in self._logged_ids:
                tl_events = self.timeline.get_timeline(interaction.interaction_id)
                self.logger.log_interaction(interaction, tl_events)
                self._logged_ids.add(interaction.interaction_id)

        # Store references for external access
        context.metadata["behaviour_timeline"] = self.timeline
        context.metadata["behaviour_logger"] = self.logger
        context.metadata["behaviour_history"] = self._behaviour_history

        return context

    def finalize(self) -> None:
        """Export all accumulated logs.  Call once after video processing ends."""
        if self.output_log_path:
            self.logger.export_all(self.output_log_path)

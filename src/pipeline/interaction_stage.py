"""Pipeline stage that manages persistent person-vehicle interactions.

Wraps :class:`InteractionManager` in a :class:`Stage` so it can be composed
into the sequential processing pipeline.  On each frame the stage:

1. Reads relationships from ``context.metadata["relationships"]``.
2. Calls ``InteractionManager.update()`` to maintain interaction state.
3. Stores active interactions on ``context.interactions`` and
   ``context.metadata["interactions"]`` for downstream consumers.
"""

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.behavior.interaction_manager import InteractionManager


class InteractionStage(Stage):
    """Pipeline stage that converts spatial relationships into
    persistent interaction objects with lifecycle management."""

    def __init__(
        self,
        distance_threshold: float = 150.0,
        linger_frames: int = 10,
        end_frames: int = 30,
    ) -> None:
        self.manager = InteractionManager(
            distance_threshold=distance_threshold,
            linger_frames=linger_frames,
            end_frames=end_frames,
        )

    def process(self, context: FrameContext) -> FrameContext:
        """Update interactions and attach them to the frame context."""

        relationships = context.metadata.get("relationships", [])

        self.manager.update(relationships, context.tracks, context.frame_number)

        # Expose all non-archived interactions to downstream stages
        active = [
            i for i in self.manager.get_all()
            if i.state.value not in ("ARCHIVED",)
        ]
        context.interactions = active
        context.metadata["interactions"] = active

        # Also expose completed interactions for logging / evaluation
        context.metadata["completed_interactions"] = self.manager.get_completed()

        return context

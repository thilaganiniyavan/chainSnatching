"""Sequential processing pipeline for frame-based analysis."""

from src.core.interfaces import Stage
from src.core.models import FrameContext


class Pipeline:
    """Run a sequence of stages against a frame context."""

    def __init__(self, stages: list[Stage]) -> None:
        """Initialize the pipeline with an ordered list of stages."""

        self.stages = stages

    def run(self, context: FrameContext) -> FrameContext:
        """Execute each stage sequentially and return the final context."""

        for stage in self.stages:
            context = stage.process(context)

        return context
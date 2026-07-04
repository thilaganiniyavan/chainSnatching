"""Abstract pipeline stage interface.

Stages implement a single transformation step that accepts a
FrameContext and returns an updated FrameContext.
"""

from abc import ABC, abstractmethod

from src.core.models.frame_context import FrameContext


class Stage(ABC):
    """Base class for all pipeline stages."""

    @abstractmethod
    def process(self, context: FrameContext) -> FrameContext:
        """Process a frame context and return the updated context."""
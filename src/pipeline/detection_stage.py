"""Detection stage for the frame processing pipeline.

This stage uses the existing Detector implementation to produce detections
and an annotated frame, then stores both on the shared FrameContext.
"""

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.detection.detector import Detector


class DetectionStage(Stage):
    """Pipeline stage that runs object detection on the current frame."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.35,
    ) -> None:
        """Create a detection stage with a single reusable detector instance."""

        self.detector = Detector(
            model_path=model_path,
            confidence=confidence,
        )

    def process(self, context: FrameContext) -> FrameContext:
        """Run detection on context.frame and store results on the context."""

        annotated_frame, detections = self.detector.draw(context.frame)
        context.detections = detections
        context.metadata["annotated_frame"] = annotated_frame
        return context
"""Pipeline components for frame-based surveillance processing."""

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.pipeline.detection_stage import DetectionStage
from src.pipeline.pipeline import Pipeline
from src.pipeline.tracking_stage import TrackingStage


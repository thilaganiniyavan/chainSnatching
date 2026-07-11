from src.core.interfaces import Stage
from src.core.models import FrameContext

from src.depth.depth_estimator import DepthEstimator


class DepthStage(Stage):
    """
    Pipeline stage that computes monocular depth using
    Depth Anything V2.
    """

    def __init__(self):

        self.depth_estimator = DepthEstimator()

    def process(self, context: FrameContext) -> FrameContext:

        depth_map, depth_visualization = self.depth_estimator.estimate(
            context.frame
        )

        context.depth_map = depth_map

        print(
            f"Depth map generated: shape={depth_map.shape}, "
            f"min={depth_map.min():.2f}, "
            f"max={depth_map.max():.2f}"
        )

        context.metadata["depth_visualization"] = depth_visualization

        return context
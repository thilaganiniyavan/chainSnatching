import cv2

from src.pipeline.depth_stage import DepthStage
from src.core.models import FrameContext

frame = cv2.imread("datasets/test_images/test.avif")

context = FrameContext(
    frame=frame,
    frame_number=1,
    timestamp=0.0
)

stage = DepthStage()

context = stage.process(context)

print(context.depth_map.shape)

cv2.imshow(
    "Depth",
    context.metadata["depth_visualization"]
)

cv2.waitKey(0)
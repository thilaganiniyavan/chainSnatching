import os
import sys

import cv2
import numpy as np
import torch

# Add Depth-Anything-V2 repository to Python path
DEPTH_ANYTHING_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../Depth-Anything-V2"
    )
)

if DEPTH_ANYTHING_PATH not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_PATH)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    """
    Wrapper around DepthAnythingV2.
    Produces a dense depth map for every frame.
    """

    def __init__(
        self,
        encoder="vits",
        weights_path="weights/depth_anything_v2_vits.pth"
    ):

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        self.model = DepthAnythingV2(**model_configs[encoder])

        checkpoint = torch.load(
            weights_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)

        self.model.eval()

    def estimate(self, frame):
        """
        Returns:

        depth_map : float32 ndarray
            Relative depth values.

        depth_visualization : uint8 BGR image
            Colored depth image for visualization.
        """

        depth = self.model.infer_image(frame)

        depth = depth.astype(np.float32)

        normalized = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        colored = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_INFERNO
        )

        return depth, colored

    def get_depth(self, depth_map, x, y):

        h, w = depth_map.shape

        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))

        return float(depth_map[y, x])
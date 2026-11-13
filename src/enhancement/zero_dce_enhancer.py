"""Zero-DCE Low-Light Image Enhancer with Adaptive Luminance Gating.

Wraps the pretrained Zero-DCE (`enhance_net_nopool`) neural network in a clean,
reusable Python class. Supports automatic brightness evaluation so that
daylight/well-lit CCTV frames bypass neural enhancement with zero overhead,
while dark/nighttime frames are enhanced prior to object detection and pose
estimation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_WEIGHTS_PATH = os.path.join(
    "models", "Zero-DCE", "Zero-DCE_code", "snapshots", "Epoch99.pth"
)
ZERO_DCE_CODE_DIR = os.path.join("models", "Zero-DCE", "Zero-DCE_code")


class ZeroDCEEnhancer:
    """Adaptive low-light enhancer backed by Zero-DCE (`enhance_net_nopool`).

    Args:
        weights_path: Path to the pretrained Zero-DCE `.pth` checkpoint.
        brightness_threshold: Mean grayscale luminance threshold in `[0, 255]`.
            Frames with mean luminance below this value are considered low-light
            and enhanced via Zero-DCE when `adaptive=True`.
        adaptive: If `True` (default), only enhance frames whose mean luminance
            is strictly below `brightness_threshold`. If `False`, enhance all
            frames unconditionally.
        device: Torch device string (`"cpu"`, `"cuda"`, or `None` for auto-detect).
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        brightness_threshold: float = 85.0,
        adaptive: bool = True,
        device: str | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.brightness_threshold = float(brightness_threshold)
        self.adaptive = bool(adaptive)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Load the `enhance_net_nopool` architecture and pretrained weights."""
        code_dir = os.path.abspath(ZERO_DCE_CODE_DIR)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        import model as zero_dce_model  # type: ignore

        net = zero_dce_model.enhance_net_nopool().to(self.device)
        if os.path.exists(self.weights_path):
            state_dict = torch.load(self.weights_path, map_location=self.device)
            net.load_state_dict(state_dict)
        net.eval()
        return net

    @staticmethod
    def compute_mean_brightness(frame: np.ndarray) -> float:
        """Compute mean perceptual luminance (Y channel / grayscale) in `[0, 255]`."""
        if frame is None or frame.size == 0:
            return 255.0
        if frame.ndim == 2:
            return float(np.mean(frame))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_low_light(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return `(is_dark, mean_brightness)` for a BGR frame."""
        mean_lum = self.compute_mean_brightness(frame)
        return mean_lum < self.brightness_threshold, mean_lum

    def enhance(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float, float]:
        """Enhance a BGR frame if needed.

        Args:
            frame: Input BGR image (`uint8`, shape `(H, W, 3)`).

        Returns:
            A 4-tuple `(output_frame, was_enhanced, input_brightness, output_brightness)`.
        """
        if frame is None or frame.size == 0:
            return frame, False, 0.0, 0.0

        is_dark, input_brightness = self.is_low_light(frame)
        if self.adaptive and not is_dark:
            return frame, False, input_brightness, input_brightness

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(image_rgb)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            _, enhanced_tensor, _ = self._model(image_tensor)

        enhanced_np = (
            enhanced_tensor.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        enhanced_uint8 = (enhanced_np * 255.0).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        output_brightness = self.compute_mean_brightness(enhanced_bgr)

        return enhanced_bgr, True, input_brightness, output_brightness

"""Pipeline stage for adaptive low-light frame enhancement using Zero-DCE.

Evaluates the mean luminance of incoming CCTV frames and applies Zero-DCE
(`enhance_net_nopool`) when the frame brightness falls below the configured
threshold. Preserves the un-enhanced raw frame inside
`context.metadata["raw_frame"]` for forensic chain-of-custody verification.
"""

from __future__ import annotations

from src.core.interfaces import Stage
from src.core.models import FrameContext
from src.enhancement.zero_dce_enhancer import DEFAULT_WEIGHTS_PATH, ZeroDCEEnhancer


class LowLightEnhancementStage(Stage):
    """Pipeline stage that enhances dark/nighttime frames via Zero-DCE.

    Args:
        weights_path: Path to pretrained Zero-DCE weights (`Epoch99.pth`).
        brightness_threshold: Mean grayscale brightness threshold (`0-255`)
            below which Zero-DCE enhancement is triggered.
        adaptive: If `True`, only enhance frames below `brightness_threshold`.
            If `False`, enhance every frame.
        device: Torch device (`"cpu"`, `"cuda"`, or `None` for auto-detect).
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        brightness_threshold: float = 85.0,
        adaptive: bool = True,
        device: str | None = None,
    ) -> None:
        self.enhancer = ZeroDCEEnhancer(
            weights_path=weights_path,
            brightness_threshold=brightness_threshold,
            adaptive=adaptive,
            device=device,
        )
        self.total_frames_seen: int = 0
        self.enhanced_frames_count: int = 0

    def process(self, context: FrameContext) -> FrameContext:
        """Evaluate frame luminance and enhance low-light frames in-place."""
        self.total_frames_seen += 1

        if "raw_frame" not in context.metadata:
            context.metadata["raw_frame"] = context.frame

        enhanced_frame, was_enhanced, in_lum, out_lum = self.enhancer.enhance(
            context.frame
        )

        if was_enhanced:
            self.enhanced_frames_count += 1
            context.frame = enhanced_frame

        context.metadata["low_light_enhanced"] = was_enhanced
        context.metadata["input_brightness"] = round(in_lum, 2)
        context.metadata["output_brightness"] = round(out_lum, 2)
        context.metadata["enhanced_frame"] = enhanced_frame

        return context

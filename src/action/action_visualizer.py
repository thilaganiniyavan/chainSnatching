"""Action Visualizer & Preview Exporter.

Provides:
1. ``ActionOverlayVisualizer``: Renders action label banners, confidence progress bars,
   model backend badges, and timeline HUD panels on video frames.
2. ``ActionPreviewExporter``: Exports annotated action preview video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.action_result import ActionResult


class ActionOverlayVisualizer:
    """Renders action classification banners, confidence progress bars, and model badges on video frames.

    Args:
        font_scale: Font scale for labels.
        panel_alpha: Alpha transparency for HUD background.
    """

    def __init__(
        self,
        font_scale: float = 0.50,
        panel_alpha: float = 0.70,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        action_results: list[ActionResult],
    ) -> np.ndarray:
        """Annotate *frame* with action prediction banners and confidence bars."""
        viz = frame.copy()
        if not action_results:
            return viz

        for idx, res in enumerate(action_results):
            self._draw_action_banner(viz, res, offset_idx=idx)

        return viz

    def _draw_action_banner(
        self,
        frame: np.ndarray,
        result: ActionResult,
        offset_idx: int = 0,
    ) -> None:
        """Draw action label banner and confidence bar at top-right of frame."""
        h, w = frame.shape[:2]

        label = f"ACTION: {result.predicted_action.upper()}"
        sub_text = f"Conf: {result.action_confidence:.0%} | Model: {result.model_name} ({result.inference_time_ms:.1f}ms)"

        (tw1, _), _ = cv2.getTextSize(label, self._font, self.font_scale + 0.1, 2)
        (tw2, _), _ = cv2.getTextSize(sub_text, self._font, self.font_scale - 0.05, 1)

        max_w = max(tw1, tw2, 220)
        panel_w = max_w + 20
        panel_h = 55

        px = w - panel_w - 15
        py = 15 + offset_idx * (panel_h + 10)

        if py + panel_h > h:
            py = max(5, h - panel_h - 5)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (25, 25, 25),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        # Color coding: Green for high conf, Cyan for medium, Magenta for low/Unknown
        if result.predicted_action == "Unknown":
            banner_color = (128, 128, 128)
        elif result.action_confidence >= 0.70:
            banner_color = (0, 255, 0)
        elif result.action_confidence >= 0.40:
            banner_color = (255, 255, 0)
        else:
            banner_color = (0, 165, 255)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), banner_color, 2)

        # Text labels
        cv2.putText(
            frame,
            label,
            (px + 10, py + 22),
            self._font,
            self.font_scale + 0.1,
            banner_color,
            2,
        )
        cv2.putText(
            frame,
            sub_text,
            (px + 10, py + 42),
            self._font,
            self.font_scale - 0.05,
            (220, 220, 220),
            1,
        )

        # Confidence bar at bottom of banner
        bar_w = int((panel_w - 20) * result.action_confidence)
        cv2.rectangle(
            frame,
            (px + 10, py + panel_h - 4),
            (px + 10 + bar_w, py + panel_h - 2),
            banner_color,
            -1,
        )


class ActionPreviewExporter:
    """Exports annotated action video clips to disk."""

    @staticmethod
    def export_preview(
        action_results: list[ActionResult],
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export action annotated video clip."""
        if not action_results or not frames_dict:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = ActionOverlayVisualizer()
        f_numbers = sorted(frames_dict.keys())
        sample_frame = frames_dict[f_numbers[0]]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f_num in f_numbers:
            frame = frames_dict[f_num]
            annotated = visualizer.draw(frame, action_results)
            out.write(annotated)

        out.release()

"""Fusion Visualizer & Preview Exporter.

Provides:
1. ``FusionOverlayVisualizer``: Renders multi-modal HUD panels displaying current
   behaviour pattern, action label, fusion confidence progress bar, and evidence timeline status.
2. ``FusionPreviewExporter``: Exports multi-modal fusion preview video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.fused_interaction import FusedInteraction


class FusionOverlayVisualizer:
    """Renders multi-modal fusion HUD panels and confidence progress bars on video frames.

    Args:
        font_scale: Font scale for HUD text labels.
        panel_alpha: Alpha transparency for HUD background.
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        panel_alpha: float = 0.70,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        fusions: list[FusedInteraction],
        current_frame_number: int,
    ) -> np.ndarray:
        """Annotate *frame* with multi-modal fusion HUD panels."""
        viz = frame.copy()

        for fusion in fusions:
            if current_frame_number < fusion.start_frame or current_frame_number > fusion.end_frame:
                continue

            self._draw_fusion_hud(viz, fusion, current_frame_number)

        return viz

    def _draw_fusion_hud(
        self,
        frame: np.ndarray,
        fusion: FusedInteraction,
        current_frame_number: int,
    ) -> None:
        """Draw multi-modal fusion HUD panel at top-left of frame."""
        h, w = frame.shape[:2]

        # Extract active pattern and action for this frame index
        active_pattern = "NONE"
        active_action = "Unknown"

        for event in fusion.evidence_timeline:
            if event.get("frame") == current_frame_number:
                active_pattern = event.get("behaviour_pattern", "NONE")
                active_action = event.get("action_label", "Unknown")
                break

        lines = [
            f"FUSION: {fusion.fusion_id} | Strategy: {fusion.fusion_strategy}",
            f"Stream A (Pattern): {active_pattern} | Stream B (Action): {active_action}",
            f"Confidences: Graph={fusion.behaviour_confidence:.0%} | Action={fusion.action_confidence:.0%} | Fusion={fusion.fusion_confidence:.0%}",
        ]

        padding = 5
        line_height = int(17 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding + 8

        px = 10
        py = h - panel_h - 15

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (15, 15, 30),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (255, 191, 0), 1)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 3
            line_colour = (255, 191, 0) if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )

        # Draw fusion confidence progress bar
        bar_w = int((panel_w - 2 * padding) * fusion.fusion_confidence)
        cv2.rectangle(
            frame,
            (px + padding, py + panel_h - 6),
            (px + padding + bar_w, py + panel_h - 3),
            (255, 191, 0),
            -1,
        )


class FusionPreviewExporter:
    """Exports multi-modal fusion preview video clips to disk."""

    @staticmethod
    def export_preview(
        fusion: FusedInteraction,
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export fusion preview video clip."""
        if not frames_dict:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = FusionOverlayVisualizer()
        f_numbers = [f for f in sorted(frames_dict.keys()) if fusion.start_frame <= f <= fusion.end_frame]
        if not f_numbers:
            return

        sample_frame = frames_dict[f_numbers[0]]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f_num in f_numbers:
            frame = frames_dict[f_num]
            annotated = visualizer.draw(frame, [fusion], f_num)
            out.write(annotated)

        out.release()

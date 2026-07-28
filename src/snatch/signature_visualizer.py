"""Signature Visualizer & Preview Exporter.

Provides:
1. ``SignatureOverlayVisualizer``: Renders signature score gauges, decision badges,
   matched/missing evidence lists, and forensic HUD panels on video frames.
2. ``SignaturePreviewExporter``: Exports forensic preview video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.snatch_signature_result import SnatchSignatureResult


class SignatureOverlayVisualizer:
    """Renders forensic signature decision badges, score gauges, and evidence HUD panels on video frames.

    Args:
        font_scale: Font scale for HUD text labels.
        panel_alpha: Alpha transparency for HUD background.
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        panel_alpha: float = 0.75,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        signatures: list[SnatchSignatureResult],
    ) -> np.ndarray:
        """Annotate *frame* with forensic signature decision banners and score gauges."""
        viz = frame.copy()
        if not signatures:
            return viz

        for idx, sig in enumerate(signatures):
            self._draw_signature_hud(viz, sig, offset_idx=idx)

        return viz

    def _draw_signature_hud(
        self,
        frame: np.ndarray,
        sig: SnatchSignatureResult,
        offset_idx: int = 0,
    ) -> None:
        """Draw forensic signature HUD banner at bottom-left of frame."""
        h, w = frame.shape[:2]

        lines = [
            f"SNATCH SIGNATURE: {sig.matched_signature_name} | Score: {sig.signature_score:.2f}",
            f"Decision: {sig.decision.upper()} (Conf: {sig.confidence:.0%})",
            f"Matched: {len(sig.matched_evidence)}/6 components | Missing: {len(sig.missing_evidence)} components",
        ]

        padding = 6
        line_height = int(18 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding + 10
        panel_h = line_height * len(lines) + 2 * padding + 8

        px = 10
        py = h - panel_h - 15 - offset_idx * (panel_h + 10)
        if py < 10:
            py = max(5, 10 + offset_idx * (panel_h + 5))

        # Color coding: Red for High/Strong match, Yellow for Partial, Grey for Weak/No match
        if sig.decision in ("High Confidence Match", "Strong Match"):
            border_color = (0, 0, 255) # Red banner
            badge_color = (0, 0, 200)
        elif sig.decision == "Partial Match":
            border_color = (0, 255, 255) # Yellow banner
            badge_color = (0, 200, 200)
        else:
            border_color = (128, 128, 128)
            badge_color = (50, 50, 50)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            badge_color,
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), border_color, 2)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 4
            line_colour = (255, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )

        # Signature score progress bar
        bar_w = int((panel_w - 2 * padding) * sig.signature_score)
        cv2.rectangle(
            frame,
            (px + padding, py + panel_h - 6),
            (px + padding + bar_w, py + panel_h - 3),
            border_color,
            -1,
        )


class SignaturePreviewExporter:
    """Exports forensic signature preview video clips to disk."""

    @staticmethod
    def export_preview(
        signature: SnatchSignatureResult,
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export signature preview video clip."""
        if not frames_dict:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = SignatureOverlayVisualizer()
        f_numbers = sorted(frames_dict.keys())

        sample_frame = frames_dict[f_numbers[0]]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f_num in f_numbers:
            frame = frames_dict[f_num]
            annotated = visualizer.draw(frame, [signature])
            out.write(annotated)

        out.release()

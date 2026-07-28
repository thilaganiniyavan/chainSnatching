"""Forensic Visualizer, Thumbnail & Clip Exporter.

Provides:
1. ``ForensicOverlayVisualizer``: Renders investigator event summary cards,
   decision badges, confidence bars, and evidence timeline HUD panels on video frames.
2. ``ForensicThumbnailExporter``: Extracts keyframe image thumbnails to disk.
3. ``ForensicClipExporter``: Exports annotated event preview video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.forensic_event import ForensicEvent


class ForensicOverlayVisualizer:
    """Renders investigator event summary cards and HUD panels on video frames.

    Args:
        font_scale: Font scale for text.
        panel_alpha: Transparency alpha value.
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
        events: list[ForensicEvent],
    ) -> np.ndarray:
        """Annotate *frame* with forensic event summary cards."""
        viz = frame.copy()
        if not events:
            return viz

        for idx, event in enumerate(events):
            self._draw_event_card(viz, event, offset_idx=idx)

        return viz

    def _draw_event_card(
        self,
        frame: np.ndarray,
        event: ForensicEvent,
        offset_idx: int = 0,
    ) -> None:
        """Draw forensic event summary card on frame."""
        h, w = frame.shape[:2]

        lines = [
            f"FORENSIC RECORD: {event.event_id} | Location: {event.location}",
            f"Decision: {event.decision} (Score: {event.signature_score:.2f} | Conf: {event.confidence:.0%})",
            f"Evidence Links: Graph={event.behaviour_graph_ref != ''} | Action={event.action_timeline_ref != ''} | Fusion={event.fusion_ref != ''}",
        ]

        padding = 6
        line_height = int(18 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding + 10
        panel_h = line_height * len(lines) + 2 * padding + 6

        px = 10
        py = 10 + offset_idx * (panel_h + 10)
        if py + panel_h > h:
            py = max(5, h - panel_h - 5)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (20, 20, 40),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 215, 255), 2)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 4
            line_colour = (0, 215, 255) if i == 0 else (230, 230, 230)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )


class ForensicThumbnailExporter:
    """Extracts keyframe image thumbnails for indexed forensic events."""

    @staticmethod
    def export_thumbnail(
        event: ForensicEvent,
        keyframe: np.ndarray,
        output_path: str,
    ) -> str:
        """Extract and save keyframe image thumbnail."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, keyframe)
        event.thumbnail_path = output_path
        return output_path


class ForensicClipExporter:
    """Exports annotated forensic video clips for indexed events."""

    @staticmethod
    def export_clip(
        event: ForensicEvent,
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> str:
        """Export annotated event preview video clip."""
        if not frames_dict:
            return ""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = ForensicOverlayVisualizer()
        f_numbers = sorted(frames_dict.keys())
        sample_frame = frames_dict[f_numbers[0]]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f_num in f_numbers:
            frame = frames_dict[f_num]
            annotated = visualizer.draw(frame, [event])
            out.write(annotated)

        out.release()
        event.video_clip_path = output_path
        return output_path

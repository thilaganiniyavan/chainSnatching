"""ROI Visualizer & Clip Exporter.

Provides:
1. ``ROIOverlayVisualizer``: Renders OpenCV overlays displaying selected person ROI boxes
   (Green = Accepted, Red = Rejected), Interaction ID, Status, Quality scores, and Confidence.
2. ``ROIClipExporter``: Utility to write cropped interaction video clips to disk
   for visual inspection and debugging.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.interaction_roi import InteractionROI


class ROIOverlayVisualizer:
    """Renders ROI bounding boxes and quality HUD panels on video frames.

    Args:
        font_scale: Font scale for drawing labels.
        panel_alpha: Alpha transparency for HUD background.
        fps: Video FPS.
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        panel_alpha: float = 0.65,
        fps: float = 30.0,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self.fps = fps if fps > 0 else 30.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        rois: list[InteractionROI],
        current_frame_number: int,
    ) -> np.ndarray:
        """Annotate *frame* with ROI bounding boxes and status HUD panels."""
        viz = frame.copy()

        for roi in rois:
            if not roi.frame_index_mapping or current_frame_number not in roi.frame_index_mapping:
                continue

            idx = roi.frame_index_mapping.index(current_frame_number)
            if idx >= len(roi.expanded_bounding_boxes):
                continue

            exp_box = roi.expanded_bounding_boxes[idx]
            raw_box = roi.bounding_box_sequence[idx] if idx < len(roi.bounding_box_sequence) else exp_box

            # Green for accepted, Red/Orange for rejected
            box_colour = (0, 255, 0) if roi.is_accepted else (0, 100, 255)

            # Draw expanded ROI box (dashed/thick) and raw box
            x1, y1, x2, y2 = exp_box
            cv2.rectangle(viz, (x1, y1), (x2, y2), box_colour, 2)

            rx1, ry1, rx2, ry2 = raw_box
            cv2.rectangle(viz, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            # Draw HUD panel near top-left of expanded box
            self._draw_roi_hud(viz, roi, x1, y1, box_colour)

        return viz

    def _draw_roi_hud(
        self,
        frame: np.ndarray,
        roi: InteractionROI,
        anchor_x: int,
        anchor_y: int,
        colour: tuple[int, int, int],
    ) -> None:
        """Draw semi-transparent status HUD panel above the ROI box."""
        status = "ACCEPTED" if roi.is_accepted else f"REJECTED ({roi.rejection_reason[:20]})"
        comp = roi.quality_metrics.get("completeness", 0.0)
        stab = roi.quality_metrics.get("bounding_box_stability", 0.0)

        lines = [
            f"ROI: {roi.roi_id} | Status: {status}",
            f"Dur: {roi.frame_count}f ({roi.duration_seconds:.1f}s) | Conf: {roi.interaction_confidence:.0%}",
            f"Quality: Comp={comp:.0%} | Stab={stab:.0%}",
        ]

        padding = 5
        line_height = int(17 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        px = max(5, anchor_x)
        py = max(5, anchor_y - panel_h - 5)

        h, w = frame.shape[:2]
        if px + panel_w > w:
            px = max(5, w - panel_w - 5)
        if py < 0:
            py = anchor_y + 5

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), colour, 1)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 3
            line_colour = colour if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )


class ROIClipExporter:
    """Exports cropped interaction video clips to disk for inspection."""

    @staticmethod
    def export_clip(
        roi: InteractionROI,
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export cropped interaction clip for an accepted ROI.

        Args:
            roi: The InteractionROI object.
            frames_dict: Dict mapping frame_number -> frame (BGR np.ndarray).
            output_path: File path for output video (.avi or .mp4).
            fps: Video FPS.
        """
        if not roi.is_accepted or not roi.frame_index_mapping:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        sample_frame_idx = roi.frame_index_mapping[0]
        if sample_frame_idx not in frames_dict:
            return

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        first_box = roi.expanded_bounding_boxes[0]
        w = max(10, first_box[2] - first_box[0])
        h = max(10, first_box[3] - first_box[1])

        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for idx, f_num in enumerate(roi.frame_index_mapping):
            if f_num in frames_dict and idx < len(roi.expanded_bounding_boxes):
                frame = frames_dict[f_num]
                ex1, ey1, ex2, ey2 = roi.expanded_bounding_boxes[idx]
                crop = frame[ey1:ey2, ex1:ex2]
                if crop.size > 0:
                    resized = cv2.resize(crop, (w, h))
                    out.write(resized)

        out.release()

"""Skeleton Sequence Visualizer & Preview Exporter.

Provides:
1. ``SkeletonSequenceVisualizer``: Renders motion trails (historical joint trajectory lines),
   confidence heatmaps, joint positions, and timeline completeness HUD panels on video frames.
2. ``SequencePreviewExporter``: Exports sequence preview video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.skeleton_sequence import SkeletonSequence


class SkeletonSequenceVisualizer:
    """Renders skeleton motion trails, joint confidence heatmaps, and timeline HUD panels.

    Args:
        trail_length: Max historical frame count for motion trail lines.
        font_scale: Font scale for HUD labels.
        panel_alpha: Alpha transparency for HUD background.
    """

    def __init__(
        self,
        trail_length: int = 15,
        font_scale: float = 0.45,
        panel_alpha: float = 0.65,
    ) -> None:
        self.trail_length = trail_length
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        sequences: list[SkeletonSequence],
        current_frame_number: int,
    ) -> np.ndarray:
        """Annotate *frame* with motion trails, joint confidence heatmaps, and HUD panels."""
        viz = frame.copy()

        for seq in sequences:
            if current_frame_number not in seq.frame_indices:
                continue

            idx = seq.frame_indices.index(current_frame_number)
            tensor = seq.skeleton_tensor # (T, V, 4)

            if idx >= tensor.shape[0]:
                continue

            # 1. Draw motion trails for primary joints (e.g. wrists, ankles, head)
            trail_start_idx = max(0, idx - self.trail_length)
            trail_joints = [0, 9, 10, 15, 16] if seq.topology == "COCO_17" else [0, 15, 16, 27, 28]

            for j in trail_joints:
                if j < tensor.shape[1]:
                    pts = []
                    for t_idx in range(trail_start_idx, idx + 1):
                        kp = tensor[t_idx, j]
                        if kp[2] > 0.3:
                            pts.append((int(kp[0]), int(kp[1])))

                    for p_i in range(1, len(pts)):
                        alpha = p_i / max(1, len(pts))
                        color = (int(255 * alpha), int(120 * alpha), int(255 * (1 - alpha)))
                        cv2.line(viz, pts[p_i - 1], pts[p_i], color, 2)

            # 2. Draw joint confidence heatmap circles
            current_kps = tensor[idx]
            for j in range(current_kps.shape[0]):
                x, y, conf, vis = current_kps[j]
                if conf > 0.1:
                    pt = (int(x), int(y))
                    # Color map from Red (low conf) to Green (high conf)
                    hue_color = (0, int(255 * conf), int(255 * (1.0 - conf)))
                    cv2.circle(viz, pt, 4, hue_color, -1)

            # 3. Draw sequence status HUD panel
            self._draw_sequence_hud(viz, seq, idx)

        return viz

    def _draw_sequence_hud(
        self,
        frame: np.ndarray,
        sequence: SkeletonSequence,
        current_step: int,
    ) -> None:
        """Draw sequence timeline and completeness HUD panel."""
        status = "ACCEPTED" if sequence.is_accepted else f"REJECTED ({sequence.rejection_reason[:20]})"
        lines = [
            f"SEQ: {sequence.sequence_id} | Status: {status}",
            f"Step: {current_step + 1}/{sequence.frame_count} ({sequence.duration_seconds:.1f}s) | Norm: {sequence.normalization_method}",
            f"Quality: {sequence.quality_score:.2f} | Completeness: {sequence.completeness_score:.0%}",
        ]

        padding = 5
        line_height = int(17 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        px = 10
        py = 10

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        border_color = (0, 255, 0) if sequence.is_accepted else (0, 100, 255)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), border_color, 1)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 3
            line_colour = border_color if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )


class SequencePreviewExporter:
    """Exports sequence preview video clips with rendered motion trails to disk."""

    @staticmethod
    def export_preview(
        sequence: SkeletonSequence,
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export preview clip for a SkeletonSequence."""
        if not sequence.frame_indices:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = SkeletonSequenceVisualizer()
        sample_f_num = sequence.frame_indices[0]
        if sample_f_num not in frames_dict:
            return

        sample_frame = frames_dict[sample_f_num]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f_num in sequence.frame_indices:
            if f_num in frames_dict:
                frame = frames_dict[f_num]
                annotated = visualizer.draw(frame, [sequence], f_num)
                out.write(annotated)

        out.release()

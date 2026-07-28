"""Pose Visualizer & Preview Exporter.

Provides:
1. ``PoseOverlayVisualizer``: Renders 2D skeleton joint circles and limb connections
   formatted with COCO-17 or MediaPipe-33 topologies onto video frames.
2. ``PosePreviewExporter``: Utility to write rendered pose overlay video clips to disk.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.core.models.pose_result import PoseResult


# COCO 17 Limb connections (pairs of joint indices)
_COCO_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Head
    (5, 6),                                     # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (5, 11), (6, 12), (11, 12),                 # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # Legs
]

# MediaPipe 33 Limb connections
_MP_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Face
    (9, 10),                                                        # Mouth
    (11, 12),                                                       # Shoulders
    (11, 13), (13, 15), (12, 14), (14, 16),                         # Arms
    (11, 23), (12, 24), (23, 24),                                   # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),                         # Legs
]


class PoseOverlayVisualizer:
    """Renders skeleton joints, limb connections, and pose quality HUD overlays on video frames.

    Args:
        joint_radius: Circle radius for joint keypoints in pixels.
        limb_thickness: Line thickness for limb connections.
        font_scale: Font scale for HUD labels.
        panel_alpha: Alpha transparency for HUD background.
    """

    def __init__(
        self,
        joint_radius: int = 4,
        limb_thickness: int = 2,
        font_scale: float = 0.45,
        panel_alpha: float = 0.65,
    ) -> None:
        self.joint_radius = joint_radius
        self.limb_thickness = limb_thickness
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        poses: list[PoseResult],
    ) -> np.ndarray:
        """Annotate *frame* with skeleton keypoints, limbs, and pose quality HUD panels."""
        viz = frame.copy()

        for pose in poses:
            if not pose.keypoints_pixel:
                continue

            limbs = _COCO_LIMBS if pose.topology == "COCO_17" else _MP_LIMBS

            # 1. Draw limb connection lines
            for j1, j2 in limbs:
                if j1 < len(pose.keypoints_pixel) and j2 < len(pose.keypoints_pixel):
                    kp1 = pose.keypoints_pixel[j1]
                    kp2 = pose.keypoints_pixel[j2]

                    if kp1[2] > 0.3 and kp2[2] > 0.3:
                        pt1 = (int(kp1[0]), int(kp1[1]))
                        pt2 = (int(kp2[0]), int(kp2[1]))
                        cv2.line(viz, pt1, pt2, (0, 255, 255), self.limb_thickness)

            # 2. Draw joint circles
            for kp in pose.keypoints_pixel:
                x, y, conf, vis = kp
                if conf > 0.3:
                    pt = (int(x), int(y))
                    cv2.circle(viz, pt, self.joint_radius, (0, 0, 255), -1)
                    cv2.circle(viz, pt, self.joint_radius + 1, (255, 255, 255), 1)

            # 3. Draw pose quality HUD panel near top of person bounding box
            self._draw_pose_hud(viz, pose)

        return viz

    def _draw_pose_hud(
        self,
        frame: np.ndarray,
        pose: PoseResult,
    ) -> None:
        """Draw pose quality HUD panel above person bbox."""
        x1, y1, x2, y2 = pose.bbox_reference
        lines = [
            f"POSE ({pose.backend_name}): Track {pose.track_id}",
            f"Conf: {pose.overall_confidence:.0%} | Quality: {pose.quality_score:.2f}",
            f"Time: {pose.processing_time_ms:.1f}ms | Joints: {pose.num_keypoints}",
        ]

        padding = 5
        line_height = int(17 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        px = max(5, x1)
        py = max(5, y1 - panel_h - 5)

        h, w = frame.shape[:2]
        if px + panel_w > w:
            px = max(5, w - panel_w - 5)
        if py < 0:
            py = y1 + 5

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px, py),
            (px + panel_w, py + panel_h),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (0, 255, 255), 1)

        for i, line in enumerate(lines):
            ty = py + padding + line_height * (i + 1) - 3
            line_colour = (0, 255, 255) if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (px + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )


class PosePreviewExporter:
    """Exports pose overlay preview video clips to disk for inspection."""

    @staticmethod
    def export_preview(
        poses: list[PoseResult],
        frames_dict: dict[int, np.ndarray],
        output_path: str,
        fps: float = 30.0,
    ) -> None:
        """Export video clip with rendered skeleton overlays."""
        if not poses:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        visualizer = PoseOverlayVisualizer()
        sample_frame_idx = poses[0].frame_index
        if sample_frame_idx not in frames_dict:
            return

        sample_frame = frames_dict[sample_frame_idx]
        h, w = sample_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Group poses by frame_index
        poses_by_frame: dict[int, list[PoseResult]] = {}
        for p in poses:
            poses_by_frame.setdefault(p.frame_index, []).append(p)

        for f_num in sorted(poses_by_frame.keys()):
            if f_num in frames_dict:
                frame = frames_dict[f_num]
                annotated = visualizer.draw(frame, poses_by_frame[f_num])
                out.write(annotated)

        out.release()

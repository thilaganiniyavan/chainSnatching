"""Skeleton Normalizer — spatial normalization strategies for skeleton tensors.

Implements configurable normalization strategies:
- ``image``: Normalized by video frame width/height [0, 1].
- ``bbox``: Relative to bounding box bounds (x - x1)/w, (y - y1)/h.
- ``hip_centered``: Origin placed at hip midpoint, scaled by torso height/bbox diagonal.
- ``root_joint``: Origin placed at nose/root joint.
- Optional rotation normalization aligning shoulder vector horizontally.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SkeletonNormalizer:
    """Applies spatial normalization to (T, V, C) skeleton tensors.

    Args:
        method: Normalization strategy ("hip_centered", "bbox", "root_joint", "image").
        enable_rotation: Whether to rotate skeleton to align shoulders horizontally.
        image_width: Video frame width (for "image" strategy).
        image_height: Video frame height (for "image" strategy).
    """

    def __init__(
        self,
        method: str = "hip_centered",
        enable_rotation: bool = False,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> None:
        self.method = method.lower().strip()
        self.enable_rotation = enable_rotation
        self.image_width = image_width if image_width > 0 else 1920
        self.image_height = image_height if image_height > 0 else 1080

    def normalize_tensor(
        self,
        raw_tensor: np.ndarray,
        bboxes: list[tuple[int, int, int, int]] | None = None,
        topology: str = "COCO_17",
    ) -> np.ndarray:
        """Normalize a 3D skeleton tensor of shape ``(T, V, C)``.

        Args:
            raw_tensor: NumPy array of shape ``(T, V, 4)`` containing ``(x, y, conf, vis)``.
            bboxes: Optional list of bounding boxes ``(x1, y1, x2, y2)`` for each frame index.
            topology: Keypoint topology (``COCO_17`` or ``MEDIAPIPE_33``).

        Returns:
            Normalized NumPy array of shape ``(T, V, 4)``.
        """
        if raw_tensor.size == 0 or raw_tensor.ndim != 3:
            return raw_tensor

        T, V, C = raw_tensor.shape
        norm_tensor = raw_tensor.copy()

        # Hip joint indices mapping
        if topology.upper() == "COCO_17":
            left_hip_idx, right_hip_idx = 11, 12
            left_sh_idx, right_sh_idx = 5, 6
            root_idx = 0
        else: # MEDIAPIPE_33
            left_hip_idx, right_hip_idx = 23, 24
            left_sh_idx, right_sh_idx = 11, 12
            root_idx = 0

        for t in range(T):
            frame_kps = norm_tensor[t] # shape (V, C)
            bbox = bboxes[t] if bboxes and t < len(bboxes) else None

            if self.method == "image":
                frame_kps[:, 0] /= self.image_width
                frame_kps[:, 1] /= self.image_height

            elif self.method == "bbox" and bbox is not None:
                x1, y1, x2, y2 = bbox
                w_box = max(1.0, float(x2 - x1))
                h_box = max(1.0, float(y2 - y1))
                frame_kps[:, 0] = (frame_kps[:, 0] - x1) / w_box
                frame_kps[:, 1] = (frame_kps[:, 1] - y1) / h_box

            elif self.method == "root_joint":
                root_x = frame_kps[root_idx, 0]
                root_y = frame_kps[root_idx, 1]
                frame_kps[:, 0] -= root_x
                frame_kps[:, 1] -= root_y

            else: # Default: "hip_centered"
                l_hip = frame_kps[left_hip_idx, :2]
                r_hip = frame_kps[right_hip_idx, :2]

                if frame_kps[left_hip_idx, 2] > 0.1 and frame_kps[right_hip_idx, 2] > 0.1:
                    hip_center = (l_hip + r_hip) / 2.0
                else:
                    # Fallback to mean valid joints
                    valid_pts = frame_kps[frame_kps[:, 2] > 0.1, :2]
                    hip_center = np.mean(valid_pts, axis=0) if len(valid_pts) > 0 else np.array([0.0, 0.0])

                frame_kps[:, 0] -= hip_center[0]
                frame_kps[:, 1] -= hip_center[1]

                # Scale by torso height or bounding box size
                l_sh = frame_kps[left_sh_idx, :2]
                r_sh = frame_kps[right_sh_idx, :2]
                sh_center = (l_sh + r_sh) / 2.0
                torso_h = np.linalg.norm(sh_center - hip_center)

                if torso_h > 1.0:
                    frame_kps[:, :2] /= torso_h
                elif bbox is not None:
                    _, _, x2, y2 = bbox
                    diag = float(math.sqrt((x2 - bbox[0])**2 + (y2 - bbox[1])**2))
                    if diag > 1.0:
                        frame_kps[:, :2] /= diag

            # Optional rotation alignment
            if self.enable_rotation:
                l_sh = frame_kps[left_sh_idx, :2]
                r_sh = frame_kps[right_sh_idx, :2]
                dx = r_sh[0] - l_sh[0]
                dy = r_sh[1] - l_sh[1]
                angle = math.atan2(dy, dx)
                cos_a, sin_a = math.cos(-angle), math.sin(-angle)
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                frame_kps[:, :2] = np.dot(frame_kps[:, :2], rot_mat)

            norm_tensor[t] = frame_kps

        return norm_tensor

"""PyTorch Dataset & Data Augmentations for CCTV Skeleton Action Recognition.

Implements Spatial-Temporal augmentations for (C=4, T=32, V=17) skeleton graphs:
- Horizontal reflection with bilateral joint index swapping.
- Temporal speed jitter and sub-sampling.
- Gaussian coordinate perturbation and scaling.
- Random joint occlusion masking (dropout).
"""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

# COCO 17 left-right paired joint indices for bilateral reflection:
# (left_eye, right_eye) -> (1, 2)
# (left_ear, right_ear) -> (3, 4)
# (left_shoulder, right_shoulder) -> (5, 6)
# (left_elbow, right_elbow) -> (7, 8)
# (left_wrist, right_wrist) -> (9, 10)
# (left_hip, right_hip) -> (11, 12)
# (left_knee, right_knee) -> (13, 14)
# (left_ankle, right_ankle) -> (15, 16)
_COCO_SWAP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]


class SkeletonActionDataset(Dataset):
    """PyTorch Dataset for (C, T, V) skeleton action tensors.

    Args:
        data_path: Path to .npz file containing 'X' (N, C, T, V) and 'y' (N,).
        augment: Whether to apply spatial-temporal augmentations.
        indices: Optional subset indices for train/val split.
    """

    def __init__(
        self,
        data_path: str,
        augment: bool = True,
        indices: list[int] | None = None,
    ) -> None:
        raw = np.load(data_path, allow_pickle=True)
        self.X = raw["X"]  # (N, 4, 32, 17)
        self.y = raw["y"]  # (N,)
        self.taxonomy = list(raw["taxonomy"]) if "taxonomy" in raw else []
        self.augment = augment

        if indices is not None:
            self.X = self.X[indices]
            self.y = self.y[indices]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()  # (4, 32, 17)
        y = int(self.y[idx])

        if self.augment:
            x = self._apply_augmentations(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _apply_augmentations(self, tensor: np.ndarray) -> np.ndarray:
        """Apply random spatial, temporal, and occlusion transformations."""
        # 1. Random Horizontal Flip (50% probability)
        if random.random() > 0.5:
            # Invert X coordinate (channel 0)
            tensor[0, :, :] = -tensor[0, :, :]
            # Swap left-right joint indices
            for l_idx, r_idx in _COCO_SWAP_PAIRS:
                tensor[:, :, [l_idx, r_idx]] = tensor[:, :, [r_idx, l_idx]]

        # 2. Random Coordinate Scaling (0.90x to 1.10x)
        if random.random() > 0.3:
            scale = random.uniform(0.90, 1.10)
            tensor[:2, :, :] *= scale

        # 3. Gaussian Joint Perturbation / Jitter
        if random.random() > 0.3:
            noise = np.random.normal(0.0, 0.012, size=tensor[:2].shape).astype(np.float32)
            tensor[:2, :, :] += noise

        # 4. Temporal Sub-sampling / Speed Jitter (32 frames)
        if random.random() > 0.4:
            T = tensor.shape[1]
            speed_rate = random.uniform(0.85, 1.15)
            new_indices = np.linspace(0, T - 1, num=T) * speed_rate
            new_indices = np.clip(new_indices, 0, T - 1).astype(np.int32)
            tensor = tensor[:, new_indices, :]

        # 5. Joint Occlusion Dropout (mask 1-2 random joints)
        if random.random() > 0.6:
            drop_joint = random.randint(0, 16)
            tensor[:, :, drop_joint] = 0.0

        return tensor

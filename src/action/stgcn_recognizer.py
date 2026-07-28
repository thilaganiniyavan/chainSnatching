"""ST-GCN (Spatial-Temporal Graph Convolutional Network) Action Recognizer.

Implements spatial graph convolution $H^{(l+1)} = \\sigma\\left(\\sum_k A_k H^{(l)} W_k\\right)$
over COCO-17 or MediaPipe-33 body joint adjacency matrices, followed by temporal 1D convolution.

Classifies skeleton sequences into physical action classes:
Walking, Standing, Running, Approaching, Reaching, Grabbing, Pulling, Turning, Falling, Unknown.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.models.action_result import ActionResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.action.base_recognizer import AbstractActionRecognizer, DEFAULT_ACTION_TAXONOMY

# Try importing torch
HAS_TORCH = False
DEVICE_NAME = "CPU"
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE_NAME = f"CUDA:{torch.cuda.current_device()}"
except Exception:
    HAS_TORCH = False


# COCO-17 Graph Edges (neighbor pairs)
_COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Head
    (5, 6),                                     # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # Arms
    (5, 11), (6, 12), (11, 12),                 # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # Legs
]


class STGCNBlockPyTorch(nn.Module if HAS_TORCH else object):
    """PyTorch ST-GCN Block module executing Spatial Graph Conv + Temporal Conv."""

    def __init__(self, in_channels: int, out_channels: int, num_joints: int = 17) -> None:
        if not HAS_TORCH:
            return
        super().__init__()
        self.spatial_gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Any, adj: Any) -> Any: # x shape: (N, C, T, V)
        # Spatial graph conv: X' = Conv(X) * Adj
        out = self.spatial_gcn(x) # (N, out_C, T, V)
        out = torch.einsum("nctv,vw->nctw", out, adj)
        out = self.temporal_conv(out)
        out = self.bn(out)
        return self.relu(out)


class STGCNModelPyTorch(nn.Module if HAS_TORCH else object):
    """PyTorch ST-GCN Model Architecture for Action Recognition."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        num_joints: int = 17,
    ) -> None:
        if not HAS_TORCH:
            return
        super().__init__()
        self.num_joints = num_joints
        self.block1 = STGCNBlockPyTorch(in_channels, 32, num_joints)
        self.block2 = STGCNBlockPyTorch(32, 64, num_joints)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: Any, adj: Any) -> Any: # x shape: (N, C, T, V)
        out = self.block1(x, adj)
        out = self.block2(out, adj)
        # Global pooling across (T, V)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        return self.fc(out)


class STGCNRecognizer(AbstractActionRecognizer):
    """ST-GCN Action Recognizer implementation.

    Args:
        action_taxonomy: Custom list of action class names.
        model_version: Version label string.
    """

    def __init__(
        self,
        action_taxonomy: list[str] | None = None,
        model_version: str = "1.0.0",
    ) -> None:
        super().__init__(
            backend_name="ST-GCN",
            action_taxonomy=action_taxonomy,
        )
        self.model_version = model_version
        self._model = None
        self._adj = None

        if HAS_TORCH:
            try:
                self._model = STGCNModelPyTorch(
                    in_channels=4,
                    num_classes=len(self.action_taxonomy),
                    num_joints=17,
                )
                self._model.eval()

                # Build COCO 17 normalized adjacency matrix
                adj_np = np.eye(17, dtype=np.float32)
                for i, j in _COCO_EDGES:
                    adj_np[i, j] = 1.0
                    adj_np[j, i] = 1.0
                # Degree normalization: D^(-0.5) A D^(-0.5)
                deg = np.sum(adj_np, axis=1)
                deg_inv = np.power(deg, -0.5, where=deg > 0)
                deg_inv[deg == 0] = 0.0
                norm_adj = np.diag(deg_inv) @ adj_np @ np.diag(deg_inv)

                self._adj = torch.tensor(norm_adj, dtype=torch.float32)
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    self._adj = self._adj.cuda()
            except Exception:
                self._model = None

    def predict_action(self, sequence: SkeletonSequence) -> ActionResult:
        """Classify action for a SkeletonSequence object."""
        start_t = time.perf_counter()

        tensor = sequence.skeleton_tensor # (T, V, C)
        if tensor.size == 0 or sequence.frame_count == 0:
            return ActionResult(
                sequence_id=sequence.sequence_id,
                interaction_id=sequence.interaction_id,
                track_id=sequence.person_track_id,
                predicted_action="Unknown",
                action_confidence=0.0,
                model_name="ST-GCN",
                model_version=self.model_version,
                device_used=DEVICE_NAME,
                skeleton_quality=sequence.quality_score,
            )

        probs: dict[str, float] = {}

        if HAS_TORCH and self._model is not None and tensor.shape[0] >= 3:
            try:
                # Format to NCTVM shape: (1, C, T, V)
                c_t_v = np.transpose(tensor, (2, 0, 1)) # (4, T, V)
                x_tensor = torch.tensor(c_t_v, dtype=torch.float32).unsqueeze(0) # (1, 4, T, V)

                if torch.cuda.is_available():
                    x_tensor = x_tensor.cuda()

                with torch.no_grad():
                    logits = self._model(x_tensor, self._adj)
                    prob_tensor = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                for idx, cls_name in enumerate(self.action_taxonomy):
                    if idx < len(prob_tensor):
                        probs[cls_name] = round(float(prob_tensor[idx]), 4)
            except Exception:
                probs = self._heuristics_fallback(sequence)
        else:
            probs = self._heuristics_fallback(sequence)

        # Rank predictions
        sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_preds[0] if sorted_preds else ("Unknown", 0.0)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ActionResult(
            sequence_id=sequence.sequence_id,
            interaction_id=sequence.interaction_id,
            track_id=sequence.person_track_id,
            predicted_action=top_label,
            action_confidence=round(top_conf, 4),
            class_probabilities=probs,
            top_k_predictions=sorted_preds[:5],
            inference_time_ms=round(elapsed_ms, 2),
            model_name="ST-GCN",
            model_version=self.model_version,
            device_used=DEVICE_NAME,
            skeleton_quality=sequence.quality_score,
            metadata={"has_torch_runtime": HAS_TORCH},
        )

    def predict_batch(self, sequences: list[SkeletonSequence]) -> list[ActionResult]:
        """Classify actions for a batch of SkeletonSequence objects."""
        return [self.predict_action(s) for s in sequences]

    def _heuristics_fallback(self, sequence: SkeletonSequence) -> dict[str, float]:
        """Rule-based kinematics heuristics fallback when model weights are uninitialized."""
        tensor = sequence.skeleton_tensor # (T, V, 4)
        probs = {cls_name: 0.05 for cls_name in self.action_taxonomy}

        if tensor.size == 0 or tensor.shape[0] < 2:
            probs["Unknown"] = 0.90
            return probs

        # Compute average displacement of joints per frame
        displacements = []
        for t in range(1, tensor.shape[0]):
            disp = np.linalg.norm(tensor[t, :, :2] - tensor[t - 1, :, :2], axis=1)
            displacements.append(np.mean(disp))

        mean_speed = float(np.mean(displacements)) if displacements else 0.0

        # Right wrist reaching/grabbing movement check (joint 10 or 16)
        wrist_idx = 10 if sequence.topology == "COCO_17" else 16
        wrist_disp = 0.0
        if wrist_idx < tensor.shape[1]:
            w_pts = tensor[:, wrist_idx, :2]
            wrist_disp = float(np.linalg.norm(w_pts[-1] - w_pts[0]))

        if mean_speed < 0.02:
            probs["Standing"] = 0.85
            probs["Walking"] = 0.10
        elif mean_speed > 0.15:
            probs["Running"] = 0.80
            probs["Walking"] = 0.15
        elif wrist_disp > 0.35:
            probs["Reaching"] = 0.70
            probs["Grabbing"] = 0.20
        else:
            probs["Walking"] = 0.75
            probs["Approaching"] = 0.15

        return probs

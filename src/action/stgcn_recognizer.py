"""ST-GCN (Spatial-Temporal Graph Convolutional Network) Action Recognizer.

Implements spatial graph convolution $H^{(l+1)} = \\sigma\\left(\\sum_k A_k H^{(l)} W_k\\right)$
over COCO-17 or MediaPipe-33 body joint adjacency matrices, followed by temporal 1D convolution.

Classifies skeleton sequences into physical action classes:
Walking, Standing, Running, Approaching, Reaching, Grabbing, Pulling, Turning, Falling, Unknown.
"""

from __future__ import annotations

import os
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
    """PyTorch ST-GCN Block with Spatial Graph Conv, Temporal Conv, and Residual Shortcut."""

    def __init__(self, in_channels: int, out_channels: int, num_joints: int = 17, stride: int = 1) -> None:
        if not HAS_TORCH:
            return
        super().__init__()
        self.spatial_gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2, inplace=True),
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Any, adj: Any) -> Any: # x shape: (N, C, T, V)
        res = self.residual(x)
        # Spatial graph conv: X' = Conv(X) * Adj
        g = self.spatial_gcn(x) # (N, out_C, T, V)
        g = torch.einsum("nctv,vw->nctw", g, adj)
        out = self.tcn(g) + res
        return self.relu(out)


class STGCNModelPyTorch(nn.Module if HAS_TORCH else object):
    """Deep PyTorch ST-GCN Model Architecture for Action Recognition."""

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
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.block1 = STGCNBlockPyTorch(in_channels, 64, num_joints)
        self.block2 = STGCNBlockPyTorch(64, 64, num_joints)
        self.block3 = STGCNBlockPyTorch(64, 128, num_joints, stride=2)
        self.block4 = STGCNBlockPyTorch(128, 128, num_joints)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: Any, adj: Any) -> Any: # x shape: (N, C, T, V)
        N, C, T, V = x.size()
        # Initial joint batch normalization
        x_flat = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x_norm = self.data_bn(x_flat)
        x = x_norm.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        out = self.block1(x, adj)
        out = self.block2(out, adj)
        out = self.block3(out, adj)
        out = self.block4(out, adj)
        # Global average pooling across (T, V)
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

                self.checkpoint_loaded = False
                # Load trained checkpoint if available
                weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "stgcn_coco17.pth"))
                if os.path.exists(weights_path):
                    try:
                        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                        self._model.load_state_dict(state_dict)
                        self.checkpoint_loaded = True
                    except Exception:
                        self.checkpoint_loaded = False

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
        max_prob = 0.0

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

                max_prob = float(np.max(prob_tensor)) if len(prob_tensor) > 0 else 0.0
                # If PyTorch model weights are untrained (uniform low confidence), use kinematics heuristics
                if max_prob < 0.35:
                    probs = self._heuristics_fallback(sequence)
                else:
                    for idx, cls_name in enumerate(self.action_taxonomy):
                        if idx < len(prob_tensor):
                            probs[cls_name] = round(float(prob_tensor[idx]), 4)
            except Exception:
                probs = self._heuristics_fallback(sequence)
        else:
            probs = self._heuristics_fallback(sequence)

        inference_source = "neural_network" if (HAS_TORCH and self._model is not None and max_prob >= 0.35) else "heuristics_fallback"

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
            metadata={
                "has_torch_runtime": HAS_TORCH,
                "inference_mode": inference_source,
                "neural_softmax_max": round(max_prob, 4),
                "checkpoint_loaded": getattr(self, "checkpoint_loaded", False),
            },
        )

    def predict_batch(self, sequences: list[SkeletonSequence]) -> list[ActionResult]:
        """Classify actions for a batch of SkeletonSequence objects."""
        return [self.predict_action(s) for s in sequences]

    def _heuristics_fallback(self, sequence: SkeletonSequence) -> dict[str, float]:
        """Rule-based kinematics heuristics fallback for realistic CCTV skeleton dynamics."""
        tensor = sequence.skeleton_tensor # (T, V, 4)
        probs = {cls_name: 0.05 for cls_name in self.action_taxonomy}

        if tensor.size == 0 or tensor.shape[0] < 2:
            probs["Unknown"] = 0.90
            return probs

        T = tensor.shape[0]
        topology = sequence.topology.upper()

        # Joint index resolution (COCO_17 vs MediaPipe_33)
        head_idx = 0  # 0 is Nose in both COCO_17 and MediaPipe_33
        if topology == "COCO_17":
            l_shoulder, r_shoulder = 5, 6
            l_wrist, r_wrist = 9, 10
            l_hip, r_hip = 11, 12
            l_knee, r_knee = 13, 14
            l_ankle, r_ankle = 15, 16
        else: # MediaPipe 33
            l_shoulder, r_shoulder = 11, 12
            l_wrist, r_wrist = 15, 16
            l_hip, r_hip = 23, 24
            l_knee, r_knee = 25, 26
            l_ankle, r_ankle = 27, 28

        # 1. Overall Body Displacement / Speed
        displacements = []
        for t in range(1, T):
            disp = np.linalg.norm(tensor[t, :, :2] - tensor[t - 1, :, :2], axis=1)
            displacements.append(np.mean(disp))
        mean_speed = float(np.mean(displacements)) if displacements else 0.0

        # 2. Arm & Wrist Dynamics (Elevation above hip, horizontal reach, impulse pull)
        horiz_extensions = []
        wrist_velocities = []
        arm_elevations = []
        head_wrist_dists = []
        arm_raised_flags = []

        for t in range(T):
            hip_y = (
                (float(tensor[t, l_hip, 1]) + float(tensor[t, r_hip, 1])) / 2.0
                if l_hip < tensor.shape[1] and r_hip < tensor.shape[1]
                else 0.60
            )
            sh_y = (
                (float(tensor[t, l_shoulder, 1]) + float(tensor[t, r_shoulder, 1])) / 2.0
                if l_shoulder < tensor.shape[1] and r_shoulder < tensor.shape[1]
                else 0.30
            )
            torso_h = max(0.10, abs(hip_y - sh_y))

            dx_l = (
                abs(float(tensor[t, l_wrist, 0] - tensor[t, l_shoulder, 0]))
                if l_wrist < tensor.shape[1] and l_shoulder < tensor.shape[1]
                else 0.0
            )
            dx_r = (
                abs(float(tensor[t, r_wrist, 0] - tensor[t, r_shoulder, 0]))
                if r_wrist < tensor.shape[1] and r_shoulder < tensor.shape[1]
                else 0.0
            )
            dy_l = (
                abs(float(tensor[t, l_wrist, 1] - tensor[t, l_shoulder, 1]))
                if l_wrist < tensor.shape[1] and l_shoulder < tensor.shape[1]
                else 1.0
            )
            dy_r = (
                abs(float(tensor[t, r_wrist, 1] - tensor[t, r_shoulder, 1]))
                if r_wrist < tensor.shape[1] and r_shoulder < tensor.shape[1]
                else 1.0
            )
            d_head_l = (
                float(np.linalg.norm(tensor[t, l_wrist, :2] - tensor[t, head_idx, :2]))
                if l_wrist < tensor.shape[1] and head_idx < tensor.shape[1]
                else 0.0
            )
            d_head_r = (
                float(np.linalg.norm(tensor[t, r_wrist, :2] - tensor[t, head_idx, :2]))
                if r_wrist < tensor.shape[1] and head_idx < tensor.shape[1]
                else 0.0
            )

            # Arm is actively raised if wrist is elevated to upper chest/neck level (significantly above mid-torso)
            l_raised = (float(tensor[t, l_wrist, 1]) < (sh_y + 0.25 * torso_h)) if l_wrist < tensor.shape[1] else False
            r_raised = (float(tensor[t, r_wrist, 1]) < (sh_y + 0.25 * torso_h)) if r_wrist < tensor.shape[1] else False
            arm_raised_flags.append(l_raised or r_raised)

            horiz_extensions.append(max(dx_l, dx_r))
            arm_elevations.append(min(dy_l, dy_r))
            head_wrist_dists.append(max(d_head_l, d_head_r))

        for t in range(1, T):
            v_l = (
                float(np.linalg.norm(tensor[t, l_wrist, :2] - tensor[t - 1, l_wrist, :2]))
                if l_wrist < tensor.shape[1]
                else 0.0
            )
            v_r = (
                float(np.linalg.norm(tensor[t, r_wrist, :2] - tensor[t - 1, r_wrist, :2]))
                if r_wrist < tensor.shape[1]
                else 0.0
            )
            wrist_velocities.append(max(v_l, v_r))

        max_horiz_ext = max(horiz_extensions, default=0.0)
        horiz_delta = (horiz_extensions[-1] - horiz_extensions[0]) if horiz_extensions else 0.0
        min_elevation = min(arm_elevations, default=1.0)
        max_wrist_vel = max(wrist_velocities, default=0.0)
        is_arm_raised = any(arm_raised_flags)

        # Check for pull & directed reach-retract impulse:
        # 1. Elevated reach (arm raised to upper chest/shoulder level, dx >= 0.65)
        # 2. V-profile: rapid extension outwards followed by snap-back retraction (retraction > 0.15)
        # 3. Vector reversal: dot product between reach vector and retraction vector < 0 (cos theta < -0.25)
        is_pulling = False
        is_reach_retract_impulse = False
        max_reversal_score = 0.0

        if len(horiz_extensions) >= 3 and is_arm_raised and max_horiz_ext >= 0.60:
            # Check left and right arm vectors for directional reversal
            for w_idx, s_idx in ((l_wrist, l_shoulder), (r_wrist, r_shoulder)):
                if w_idx < tensor.shape[1] and s_idx < tensor.shape[1]:
                    wrist_pts = tensor[:, w_idx, :2]
                    dists_from_shoulder = [float(np.linalg.norm(wrist_pts[t] - tensor[t, s_idx, :2])) for t in range(T)]
                    p_idx = int(np.argmax(dists_from_shoulder))
                    if 0 < p_idx < T - 1:
                        reach_vec = wrist_pts[p_idx] - wrist_pts[0]
                        retract_vec = wrist_pts[-1] - wrist_pts[p_idx]
                        norm_reach = float(np.linalg.norm(reach_vec))
                        norm_retract = float(np.linalg.norm(retract_vec))
                        if norm_reach > 0.10 and norm_retract > 0.10:
                            cos_rev = float(np.dot(reach_vec, retract_vec) / (norm_reach * norm_retract + 1e-6))
                            if cos_rev < max_reversal_score:
                                max_reversal_score = cos_rev

            peak_idx = int(np.argmax(horiz_extensions))
            if 0 < peak_idx < len(horiz_extensions) - 1:
                retraction = horiz_extensions[peak_idx] - horiz_extensions[-1]
                if retraction > 0.15 and max_wrist_vel >= 0.030:
                    is_pulling = True
                    if max_reversal_score < -0.20 or (retraction > 0.22 and max_wrist_vel >= 0.040):
                        is_reach_retract_impulse = True

        # Check for intentional reach: Arm actively raised horizontally outward towards target
        is_reaching = (
            is_arm_raised
            and max_horiz_ext >= 0.65
            and (
                (horiz_delta >= 0.18 and max_wrist_vel >= 0.035)
                or is_reach_retract_impulse
            )
            and max_wrist_vel >= 0.025
        )
        is_grabbing = (
            is_arm_raised
            and max_horiz_ext >= 0.65
            and (is_reach_retract_impulse or (horiz_delta >= 0.20 and max_wrist_vel >= 0.040))
        )

        # 3. Leg & Stride Dynamics (Running detection)
        stride_widths = []
        ankle_velocities = []
        for t in range(T):
            if l_ankle < tensor.shape[1] and r_ankle < tensor.shape[1]:
                stride = float(np.linalg.norm(tensor[t, l_ankle, :2] - tensor[t, r_ankle, :2]))
                stride_widths.append(stride)
        for t in range(1, T):
            if l_ankle < tensor.shape[1] and r_ankle < tensor.shape[1]:
                v_la = float(np.linalg.norm(tensor[t, l_ankle, :2] - tensor[t - 1, l_ankle, :2]))
                v_ra = float(np.linalg.norm(tensor[t, r_ankle, :2] - tensor[t - 1, r_ankle, :2]))
                ankle_velocities.append(max(v_la, v_ra))

        max_stride = max(stride_widths, default=0.0)
        max_ankle_vel = max(ankle_velocities, default=0.0)
        is_running = (mean_speed >= 0.08) or (max_ankle_vel >= 0.10) or (max_stride >= 0.55 and mean_speed >= 0.05)

        # 4. Torso & Vertical Dynamics (Falling / Stumbling)
        torso_y_start = (
            (tensor[0, l_hip, 1] + tensor[0, r_hip, 1]) / 2.0
            if l_hip < tensor.shape[1] and r_hip < tensor.shape[1]
            else 0.0
        )
        torso_y_end = (
            (tensor[-1, l_hip, 1] + tensor[-1, r_hip, 1]) / 2.0
            if l_hip < tensor.shape[1] and r_hip < tensor.shape[1]
            else 0.0
        )
        vertical_drop = torso_y_end - torso_y_start # In image coords, +y is downwards

        # 5. Shoulder Orientation (Turning)
        shoulder_vec_start = (
            tensor[0, r_shoulder, 0] - tensor[0, l_shoulder, 0]
            if r_shoulder < tensor.shape[1] and l_shoulder < tensor.shape[1]
            else 0.0
        )
        shoulder_vec_end = (
            tensor[-1, r_shoulder, 0] - tensor[-1, l_shoulder, 0]
            if r_shoulder < tensor.shape[1] and l_shoulder < tensor.shape[1]
            else 0.0
        )
        is_turning = (shoulder_vec_start * shoulder_vec_end < 0) and (abs(shoulder_vec_end - shoulder_vec_start) > 0.45)

        # Classification decision hierarchy:
        if vertical_drop > 0.40 * torso_h and mean_speed > 0.05:
            probs["Falling"] = 0.84
            probs["Turning"] = 0.10
        elif is_reach_retract_impulse:
            probs["Grabbing"] = 0.90
            probs["Pulling"] = 0.88
            probs["Reaching"] = 0.85
        elif is_pulling:
            probs["Pulling"] = 0.88
            probs["Grabbing"] = 0.80
            probs["Reaching"] = 0.75
        elif is_reaching:
            probs["Reaching"] = 0.82
            probs["Grabbing"] = 0.70
        elif is_grabbing:
            probs["Grabbing"] = 0.80
            probs["Reaching"] = 0.70
        elif is_turning:
            probs["Turning"] = 0.75
            probs["Walking"] = 0.15
        elif is_running:
            probs["Running"] = 0.84
            probs["Walking"] = 0.12
        elif mean_speed >= 0.015:
            probs["Walking"] = 0.80
            probs["Approaching"] = 0.12
        else:
            probs["Standing"] = 0.85
            probs["Walking"] = 0.10

        return probs

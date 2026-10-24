"""Stage 2: a learned verifier for clip candidates.

Each clip is NF frames sampled from BEFORE_S before to AFTER_S after the contact onset, cropped to the
candidate's square and resized to SIDE x SIDE. A Kinetics-400 R(2+1)D-18 network, frozen, turns the clip into a
512-d embedding; a small head (standardise -> PCA -> logistic regression), trained on actor-labelled real events,
turns that into a probability. The head is plain numpy, saved with only tensors and primitives.

A model fine-tuned end to end on a GPU (finetune.py) is the same network with its own weights and a linear output;
Verifier loads either kind, and both run through OpenVINO on a CPU at the same cost. A fine-tuned Swin3D-T
(finetune.py --arch swin3d_t) runs through PyTorch; on a GPU, so can its multi-view scoring (Verifier.shifts/flip).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

NF = 12              # frames per clip
SIDE = 112           # crop size fed to the network
BEFORE_S = 1.5       # clip starts this long before the onset ...
AFTER_S = 2.5        # ... and ends this long after it
MODEL_FORMAT = "snatchguard-verifier-v1"
FT_FORMAT = "snatchguard-verifier-ft-v1"     # finetune.py
_MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], np.float32)


def sample_times(t0: float, shift: float = 0.0) -> np.ndarray:
    return np.linspace(t0 + shift - BEFORE_S, t0 + shift + AFTER_S, NF)


def cut(img_bgr: np.ndarray, crop: tuple[float, float, float], scale: float = 1.0, flip: bool = False,
        side: int = SIDE) -> np.ndarray:
    """Square crop (x0, y0, side) given in analysis-frame pixels; `scale` maps those to `img_bgr` pixels.
    Parts outside the frame are black. Returns side x side RGB uint8."""
    x0, y0, sd = (v * scale for v in crop)
    X0, Y0, SD = int(round(x0)), int(round(y0)), max(int(round(sd)), 8)
    h, w = img_bgr.shape[:2]
    out = np.zeros((SD, SD, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X0 + SD, w), min(Y0 + SD, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = img_bgr[sy0:sy1, sx0:sx1]
    out = cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return np.ascontiguousarray(out[:, ::-1] if flip else out)


class Encoder:
    """Frozen Kinetics-400 R(2+1)D-18 trunk: (N, NF, SIDE, SIDE, 3) RGB uint8 -> (N, 512) float32.
    On CPU it runs through OpenVINO when available (same weights, same numbers, ~1.5x faster), else PyTorch."""

    def __init__(self, threads: int = 2, device: str = "cpu", backend: str = "auto") -> None:
        import torch
        torch.set_num_threads(max(1, int(threads)))
        self.torch = torch
        self.device = device
        self.net = self.ov = None
        if backend in ("auto", "openvino") and device == "cpu":
            try:
                self.ov = self._openvino(int(threads))
            except Exception:
                if backend == "openvino":
                    raise
        if self.ov is None:
            self.net = self._torch_net().to(device)
        self.backend = "openvino" if self.ov is not None else "torch"

    def _torch_net(self):
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        net.fc = self.torch.nn.Identity()
        return net.eval()

    def _openvino(self, threads: int):
        """The same network as OpenVINO IR, converted once (~10 s) and kept next to the PyTorch weights."""
        import openvino as ov
        from torchvision.models.video import R2Plus1D_18_Weights
        name = Path(R2Plus1D_18_Weights.KINETICS400_V1.url).stem          # names the exact weights
        ir = Path(self.torch.hub.get_dir()) / "checkpoints" / f"{name}.trunk.xml"
        if not ir.exists():
            model = ov.convert_model(self._torch_net(), example_input=self.torch.zeros(1, 3, NF, SIDE, SIDE))
            ir.parent.mkdir(parents=True, exist_ok=True)
            tmp = ir.with_name(f".{name}.{os.getpid()}.xml")
            ov.save_model(model, tmp, compress_to_fp16=False)
            os.replace(tmp.with_suffix(".bin"), ir.with_suffix(".bin"))
            os.replace(tmp, ir)
        core = ov.Core()
        return core.compile_model(core.read_model(ir), "CPU",
                                  {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"})

    def encode(self, clips: np.ndarray, batch: int = 8) -> np.ndarray:
        if len(clips) == 0:
            return np.zeros((0, 512), np.float32)
        out = []
        for b0 in range(0, len(clips), batch):
            x = np.ascontiguousarray(((clips[b0:b0 + batch].astype(np.float32) / 255.0 - _MEAN) / _STD)
                                     .transpose(0, 4, 1, 2, 3))
            if self.ov is not None:
                out.append(np.asarray(self.ov(x)[0]))
                continue
            with self.torch.inference_mode():
                out.append(self.net(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


@dataclass
class Head:
    scaler_mean: np.ndarray      # (D,)
    scaler_scale: np.ndarray     # (D,)
    pca_mean: np.ndarray         # (D,) mean of the standardised embeddings
    pca_components: np.ndarray   # (k, D)
    coef: np.ndarray             # (k,)
    intercept: float
    threshold: float = 0.5       # alert threshold, chosen on training data at a false-alarm budget
    meta: dict = field(default_factory=dict)

    def decision(self, emb: np.ndarray) -> np.ndarray:
        z = (emb.astype(np.float32) - self.scaler_mean) / self.scaler_scale
        return ((z - self.pca_mean) @ self.pca_components.T) @ self.coef + self.intercept

    def proba(self, emb: np.ndarray) -> np.ndarray:
        if len(emb) == 0:
            return np.zeros(0, np.float32)
        return (1.0 / (1.0 + np.exp(-self.decision(emb)))).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = {k: torch.from_numpy(np.ascontiguousarray(getattr(self, k), dtype=np.float32))
               for k in ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "coef")}
        tmp = path.with_suffix(".tmp")
        torch.save({"format": MODEL_FORMAT, "intercept": float(self.intercept), "threshold": float(self.threshold),
                    "meta": self.meta, "clip": {"nf": NF, "side": SIDE, "before_s": BEFORE_S, "after_s": AFTER_S},
                    **arr}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "Head":
        import torch
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        if d.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path}: not a {MODEL_FORMAT} model")
        clip = d.get("clip", {})
        if (clip.get("nf"), clip.get("side")) != (NF, SIDE):
            raise ValueError(f"{path}: trained for {clip} clips, this code makes {NF} x {SIDE}")
        return cls(*(d[k].numpy() for k in ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "coef")),
                   d["intercept"], d["threshold"], d.get("meta", {}))

    @classmethod
    def fit(cls, emb: np.ndarray, y: np.ndarray, n_components: int = 32, C: float = 0.05,
            sample_weight: np.ndarray | None = None) -> "Head":
        """Standardise -> PCA -> class-balanced L2 logistic regression (scikit-learn, training only)."""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        mean = emb.mean(0)
        scale = emb.std(0) + 1e-6
        z = (emb - mean) / scale
        k = int(min(n_components, len(z) - 1, z.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(z)
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        lr.fit(pca.transform(z), y, sample_weight=sample_weight)
        return cls(mean.astype(np.float32), scale.astype(np.float32), pca.mean_.astype(np.float32),
                   pca.components_.astype(np.float32), lr.coef_[0].astype(np.float32), float(lr.intercept_[0]))


class FineTuned(Encoder):
    """A network fine-tuned by finetune.py: (N, NF, SIDE, SIDE, 3) RGB uint8 -> (N,) logits.
    The OpenVINO copy is converted once and kept next to the model file, named by a hash of the weights."""

    def __init__(self, ckpt: dict, path: Path, threads: int = 2, device: str = "cpu", backend: str = "auto") -> None:
        self.ckpt, self.path = ckpt, Path(path)
        super().__init__(threads, device, backend)

    def _torch_net(self):
        from snatchguard.verifier.finetune import build_net
        net = build_net(pretrained=False, arch=self.ckpt.get("arch", "r2plus1d_18"))
        net.load_state_dict(self.ckpt["state_dict"])
        return net.eval()

    def _openvino(self, threads: int):
        import hashlib
        import openvino as ov
        h = hashlib.md5(self.path.read_bytes()).hexdigest()[:12]
        ir = self.path.with_name(f"{self.path.stem}.{h}.xml")
        if not ir.exists():
            model = ov.convert_model(self._torch_net(), example_input=self.torch.zeros(1, 3, NF, SIDE, SIDE))
            tmp = ir.with_name(f".{ir.stem}.{os.getpid()}.xml")
            ov.save_model(model, tmp, compress_to_fp16=False)
            os.replace(tmp.with_suffix(".bin"), ir.with_suffix(".bin"))
            os.replace(tmp, ir)
        core = ov.Core()
        return core.compile_model(core.read_model(ir), "CPU",
                                  {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"})

    def encode(self, clips: np.ndarray, batch: int = 8) -> np.ndarray:
        if len(clips) == 0:
            return np.zeros(0, np.float32)
        return super().encode(clips, batch).reshape(-1)


class Verifier:
    """`shifts` (s): the clip is cut once per shift of its window (the live stage does this); with `flip` each cut is
    also scored mirrored; the score is the mean logit of all views. tta: True / False / "auto" (what `finetune
    select` chose for the model); views need a model trained with them, as their threshold is fixed with them."""

    def __init__(self, model_path: str | Path, threads: int = 2, device: str = "cpu",
                 encoder: Encoder | None = None, tta: bool | str = "auto") -> None:
        import torch
        ckpt = torch.load(Path(model_path), map_location="cpu", weights_only=True)
        self.shifts, self.flip = (0.0,), False
        self.threshold_confirm: float | None = None       # CONFIRMED alerts at or above this (None: SUSPECT only)
        if ckpt.get("format") == FT_FORMAT:
            clip = ckpt.get("clip", {})
            if (clip.get("nf"), clip.get("side")) != (NF, SIDE):
                raise ValueError(f"{model_path}: trained for {clip} clips, this code makes {NF} x {SIDE}")
            self.head = None
            self.threshold = float(ckpt["threshold"])
            self.threshold_confirm = ckpt.get("threshold_confirm")
            views = ckpt.get("tta")
            want = bool(ckpt.get("tta_default", False)) if tta == "auto" else bool(tta)
            if want and views is None:
                log.warning("%s has no multi-view threshold: scoring the live view only", model_path)
            elif want:
                self.threshold, self.flip = float(views["threshold"]), bool(views["flip"])
                self.threshold_confirm = views.get("threshold_confirm")
                self.shifts = tuple(float(x) for x in views["shifts_s"])
            self.encoder = FineTuned(ckpt, Path(model_path), threads, device)
        else:
            self.head = Head.load(model_path)
            self.threshold = self.head.threshold
            self.encoder = encoder or Encoder(threads, device)
        self.backend = self.encoder.backend

    def score(self, clips: np.ndarray) -> np.ndarray:
        """(N, NF, SIDE, SIDE, 3) RGB uint8, or (N, len(shifts), NF, SIDE, SIDE, 3) -> (N,) scores."""
        clips = np.asarray(clips)
        if clips.ndim == 5:
            clips = clips[:, None]
        n, v = clips.shape[:2]
        flat = clips.reshape(n * v, *clips.shape[2:])
        if self.head is not None:
            return self.head.proba(self.encoder.encode(flat)).reshape(n, v).mean(1).astype(np.float32)
        if self.flip:
            flat = np.concatenate([flat, flat[:, :, :, ::-1]])
        z = self.encoder.encode(flat).reshape(-1, n, v).mean(axis=(0, 2))
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

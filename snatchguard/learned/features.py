"""Per-frame features (frozen ImageNet CNN over whole frame + overlapping tiles, plus frame-difference motion)
and the window features the head is trained on.

Window features use only how the picture *changes* inside the window (spread, jumps and drift of the
embeddings, motion energy), never the window's average appearance: the training clips come from very
different cameras than the normal footage, and absolute appearance would let a model learn the camera
instead of the event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

FEATURE_VERSION = 1
N_MOTION = 3
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FeatureSpec:
    backbone: str = "mobilenet_v3_large"
    input_px: int = 160
    max_side: int = 416          # every frame is first downscaled to this long side: equalises HD and low-res sources
    sample_fps: float = 4.0
    tiles: int = 2               # tiles x tiles overlapping crops, plus the whole frame
    tile_frac: float = 0.6
    motion_px: int = 64

    @property
    def n_views(self) -> int:
        return 1 + self.tiles * self.tiles

    @property
    def tag(self) -> str:
        return (f"{self.backbone}-{self.input_px}px-s{self.max_side}-f{self.sample_fps:g}-t{self.tiles}"
                f"-v{FEATURE_VERSION}")

    def to_dict(self) -> dict:
        return asdict(self)


def standardise(img: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    h, w = img.shape[:2]
    s = spec.max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def views(img: np.ndarray, spec: FeatureSpec) -> list[np.ndarray]:
    """Whole frame, then tiles row by row. Tiles zoom in on small, distant people."""
    h, w = img.shape[:2]
    out = [img]
    if spec.tiles > 1:
        tw, th = max(1, int(w * spec.tile_frac)), max(1, int(h * spec.tile_frac))
        for i in range(spec.tiles):
            y0 = round(i * (h - th) / (spec.tiles - 1))
            for j in range(spec.tiles):
                x0 = round(j * (w - tw) / (spec.tiles - 1))
                out.append(img[y0:y0 + th, x0:x0 + tw])
    return out


def resolve_device(device: str | int | None) -> str:
    d = str(device if device is not None else "cpu")
    return f"cuda:{d}" if d.isdigit() else d


class Backbone:
    """Frozen torchvision ImageNet classifier used as an embedding (global-average-pooled last feature map)."""

    def __init__(self, name: str, device: str = "cpu", threads: int | None = 2) -> None:
        import torch
        from torchvision.models import get_model, get_model_weights

        if threads and resolve_device(device) == "cpu":
            torch.set_num_threads(threads)
        self.torch = torch
        self.device = torch.device(resolve_device(device))
        model = get_model(name, weights=get_model_weights(name).DEFAULT).eval()
        if hasattr(model, "features") and hasattr(model, "avgpool"):        # mobilenet / efficientnet
            self.net = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten())
        else:                                                                # resnet / regnet: drop the classifier
            model.fc = torch.nn.Identity()
            self.net = model
        self.net = self.net.to(self.device)
        with torch.inference_mode():
            self.dim = int(self.net(torch.zeros(1, 3, 64, 64, device=self.device)).shape[1])

    def embed(self, crops: list[np.ndarray], px: int) -> np.ndarray:
        batch = np.stack([cv2.resize(c, (px, px), interpolation=cv2.INTER_AREA) for c in crops])
        batch = (batch[..., ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD   # BGR -> RGB, ImageNet norm
        x = self.torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(self.device)
        with self.torch.inference_mode():
            return self.net(x).float().cpu().numpy()


def motion_features(gray: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """Mean change, fraction of changed pixels, strongest 8x8-block change between consecutive samples."""
    if prev is None:
        return np.zeros(N_MOTION, dtype=np.float32)
    d = np.abs(gray - prev) / 255.0
    n = d.shape[0] // 8
    blocks = d[:n * 8, :n * 8].reshape(8, n, 8, n).mean(axis=(1, 3))
    return np.array([d.mean(), (d > 0.1).mean(), blocks.max()], dtype=np.float32)


class FrameFeaturizer:
    """One sampled frame -> (views x D embedding, views x 3 motion). Keeps the previous frame for motion."""

    def __init__(self, spec: FeatureSpec, backbone: Backbone) -> None:
        self.spec = spec
        self.backbone = backbone
        self.prev: list[np.ndarray] | None = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crops = views(standardise(img, self.spec), self.spec)
        px = self.spec.motion_px
        grays = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (px, px), interpolation=cv2.INTER_AREA)
                 .astype(np.float32) for c in crops]
        motion = np.stack([motion_features(g, None if self.prev is None else self.prev[i])
                           for i, g in enumerate(grays)])
        self.prev = grays
        return self.backbone.embed(crops, self.spec.input_px), motion


class Sampler:
    """Picks frames at `fps` on the frame's own clock, whatever rate frames arrive at."""

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self.next_t: float | None = None

    def reset(self) -> None:
        self.next_t = None

    def take(self, t: float) -> bool:
        if self.next_t is not None and t < self.next_t - 1e-6:
            return False
        self.next_t = t + self.dt if self.next_t is None else max(self.next_t + self.dt, t + 0.5 * self.dt)
        return True


# ---- windows ---------------------------------------------------------------------------------------

def window_spans(times: np.ndarray, window_s: float, stride_s: float, final: bool = True) -> list[tuple[float, float]]:
    """Windows [s, s + window_s) every stride_s from the first sample. A window exists once samples cover it.
    `final`: the stream has ended, so a clip shorter than one window still gets one window over all of it."""
    if len(times) < 2:
        return []
    t0, t_last = float(times[0]), float(times[-1])
    covered = t_last + 0.5 * (times[-1] - times[-2])       # the last sample stands for half a step after it
    spans, s = [], t0
    while s + window_s <= covered + 1e-6:
        spans.append((round(s, 3), round(s + window_s, 3)))
        s += stride_s
    if not spans and final:
        spans.append((round(t0, 3), round(covered, 3)))
    return spans


def window_feature(z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """z: (T, V, k) projected embeddings, m: (T, V, 3) motion -> one vector.
    Per view: spread, largest jump, and drift of the embedding; step size; motion max / mean / rise.
    Views are summarised as the whole frame, the max over tiles and the mean over tiles."""
    t, v, k = z.shape
    if t < 2:
        z = np.concatenate([z, z], 0)
        m = np.concatenate([m, m], 0)
    dz = np.diff(z, axis=0)
    step = np.linalg.norm(dz, axis=2)                                   # (T-1, V)
    per_view = np.concatenate([
        z.std(0), np.abs(dz).max(0), np.abs(z[-1] - z[0]),
        np.stack([step.max(0), step.mean(0)], 1),
        m.max(0), m.mean(0), np.diff(m, axis=0).max(0),
    ], 1)                                                               # (V, 3k + 2 + 9)
    if v == 1:
        return np.concatenate([per_view[0], per_view[0], per_view[0]]).astype(np.float32)
    tiles = per_view[1:]
    return np.concatenate([per_view[0], tiles.max(0), tiles.mean(0)]).astype(np.float32)


def window_dim(k: int) -> int:
    return 3 * (3 * k + 2 + 3 * N_MOTION)


def windows_for(times: np.ndarray, z: np.ndarray, m: np.ndarray, window_s: float, stride_s: float,
                final: bool = True) -> tuple[list[tuple[float, float]], np.ndarray]:
    """All windows of one video: spans and their features (n, window_dim)."""
    spans = window_spans(times, window_s, stride_s, final)
    feats = []
    for s, e in spans:
        idx = np.nonzero((times >= s - 1e-6) & (times < e - 1e-6))[0]
        if len(idx) == 0:
            idx = np.array([np.searchsorted(times, s).clip(0, len(times) - 1)])
        feats.append(window_feature(z[idx], m[idx]))
    k = z.shape[2]
    return spans, (np.stack(feats) if feats else np.zeros((0, window_dim(k)), np.float32))

"""Live and offline snatch scoring from pixels: frozen CNN features -> window features -> calibrated head.

    scorer = SnatchScorer("models/learned/snatch_scorer.pt")
    for img, t in frames:
        for w in scorer.push(img, t):      # WindowScore(t_start, t_end, score)
            ...

The head is plain numpy (PCA -> standardise -> logistic regression -> Platt calibration), so inference
needs torch/torchvision for the backbone only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from snatchguard.learned.features import Backbone, FeatureSpec, FrameFeaturizer, Sampler, window_feature

MODEL_FORMAT = "snatchguard-learned-v1"


@dataclass
class WindowScore:
    t_start: float
    t_end: float
    score: float


@dataclass
class Head:
    """Everything after the backbone. Arrays are float32 numpy."""

    spec: FeatureSpec
    window_s: float
    stride_s: float
    pca_mean: np.ndarray        # (D,)
    pca_components: np.ndarray  # (k, D)
    scaler_mean: np.ndarray     # (F,)
    scaler_scale: np.ndarray    # (F,)
    coef: np.ndarray            # (F,)
    intercept: float
    platt_a: float
    platt_b: float
    threshold: float = 0.5      # model-only alert threshold (chosen on training data only)
    min_run: int = 1            # consecutive windows over threshold needed for a model-only alert
    meta: dict = field(default_factory=dict)

    def project(self, emb: np.ndarray) -> np.ndarray:
        """(..., D) embeddings -> (..., k)."""
        return (emb.astype(np.float32) - self.pca_mean) @ self.pca_components.T

    def decision(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.scaler_mean) / self.scaler_scale) @ self.coef + self.intercept

    def proba(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0, np.float32)
        return 1.0 / (1.0 + np.exp(-(self.platt_a * self.decision(feats) + self.platt_b)))

    # ---- (de)serialisation: only tensors and primitives, so torch.load(weights_only=True) works ------
    def save(self, path: str | Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = {k: torch.from_numpy(np.ascontiguousarray(getattr(self, k), dtype=np.float32))
               for k in ("pca_mean", "pca_components", "scaler_mean", "scaler_scale", "coef")}
        torch.save({"format": MODEL_FORMAT, "spec": self.spec.to_dict(), "window_s": self.window_s,
                    "stride_s": self.stride_s, "intercept": float(self.intercept), "platt_a": float(self.platt_a),
                    "platt_b": float(self.platt_b), "threshold": float(self.threshold), "min_run": int(self.min_run),
                    "meta": self.meta, **arr}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "Head":
        import torch
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        if d.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path}: not a {MODEL_FORMAT} model")
        return cls(FeatureSpec(**d["spec"]), d["window_s"], d["stride_s"],
                   *(d[k].numpy() for k in ("pca_mean", "pca_components", "scaler_mean", "scaler_scale", "coef")),
                   d["intercept"], d["platt_a"], d["platt_b"], d["threshold"], d["min_run"], d.get("meta", {}))


class SnatchScorer:
    def __init__(self, model_path: str | Path, device: str = "cpu", threads: int = 2,
                 backbone: Backbone | None = None) -> None:
        self.head = Head.load(model_path)
        self.spec = self.head.spec
        self.window_s = self.head.window_s
        self.stride_s = self.head.stride_s
        self.threshold = self.head.threshold
        self.featurizer = FrameFeaturizer(self.spec, backbone or Backbone(self.spec.backbone, device, threads))
        self.sampler = Sampler(self.spec.sample_fps)
        keep = int(np.ceil((self.window_s + self.stride_s) * self.spec.sample_fps)) + 4
        self._t: deque[float] = deque(maxlen=keep)
        self._z: deque[np.ndarray] = deque(maxlen=keep)
        self._m: deque[np.ndarray] = deque(maxlen=keep)
        self._next_start: float | None = None
        self._emitted = False

    def reset(self) -> None:
        self.featurizer.reset()
        self.sampler.reset()
        self._t.clear()
        self._z.clear()
        self._m.clear()
        self._next_start = None
        self._emitted = False

    def push(self, img_bgr: np.ndarray, t: float) -> list[WindowScore]:
        if not self.sampler.take(t):
            return []
        emb, motion = self.featurizer(img_bgr)
        self._t.append(float(t))
        self._z.append(self.head.project(emb))
        self._m.append(motion)
        if self._next_start is None:
            self._next_start = float(t)
        return self._emit(final=False)

    def flush(self) -> list[WindowScore]:
        """End of a recording: score the last window if the clip was shorter than one window."""
        return self._emit(final=True)

    def _emit(self, final: bool) -> list[WindowScore]:
        """Same grid as features.window_spans: starts at the first sample, steps by stride_s."""
        if len(self._t) < 2:
            return []
        times = np.array(self._t)
        covered = times[-1] + 0.5 * (times[-1] - times[-2])
        out = []
        while self._next_start + self.window_s <= covered + 1e-6:
            s = self._next_start
            out.append(self._score(times, round(s, 3), round(s + self.window_s, 3)))
            self._next_start = s + self.stride_s
            self._emitted = True
        if final and not self._emitted:
            out.append(self._score(times, round(float(times[0]), 3), round(float(covered), 3)))
            self._emitted = True
        return out

    def _score(self, times: np.ndarray, s: float, e: float) -> WindowScore:
        idx = np.nonzero((times >= s - 1e-6) & (times < e - 1e-6))[0]
        z = np.stack([self._z[i] for i in idx])
        m = np.stack([self._m[i] for i in idx])
        p = float(self.head.proba(window_feature(z, m)[None])[0])
        return WindowScore(round(s, 3), round(e, 3), p)


def iter_video(path: str | Path, sample_fps: float, max_seconds: float | None = None):
    """(t, frame) at about `sample_fps`, decoding only what is needed."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sampler = Sampler(sample_fps)
    idx = 0
    try:
        while cap.grab():
            t = idx / src_fps
            idx += 1
            if max_seconds is not None and t > max_seconds:
                break
            if not sampler.take(t):
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            yield t, img
    finally:
        cap.release()


def score_video(path: str | Path, model_path: str | Path, device: str = "cpu",
                scorer: SnatchScorer | None = None) -> list[WindowScore]:
    scorer = scorer or SnatchScorer(model_path, device)
    scorer.reset()
    out: list[WindowScore] = []
    for t, img in iter_video(path, scorer.spec.sample_fps):
        out += scorer.push(img, t)
    return out + scorer.flush()


def alerts_from_windows(windows: list, threshold: float, min_run: int = 1) -> list[dict]:
    """Model-only alerts: runs of consecutive windows with score >= threshold, merged into one alert each.
    `windows`: WindowScore objects or [t_start, t_end, score] triples, in time order."""
    rows = [(w.t_start, w.t_end, w.score) if isinstance(w, WindowScore) else tuple(w) for w in windows]
    alerts, run = [], []

    def close():
        if len(run) >= min_run:
            alerts.append({"level": "SUSPECT", "source": "model", "contact_start": run[0][0],
                           "contact_end": run[-1][1], "decided_at": run[min_run - 1][1],
                           "score": round(max(r[2] for r in run), 4)})

    for r in rows:
        if r[2] >= threshold:
            run.append(r)
        elif run:
            close()
            run = []
    if run:
        close()
    return alerts

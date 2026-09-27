"""Camera tamper / failure detection.

A detector that silently goes blind is worse than no detector, so every camera
is checked once a second for:
  * blackout  - lens covered, camera dead, IR failed (image nearly black)
  * blur      - sprayed / smeared / defocused lens (sharpness collapses vs. baseline)
  * moved     - camera turned away (scene no longer matches its learned reference)
  * frozen    - the same frame keeps arriving (encoder / network stall)
A condition must persist before it is reported, and clears with hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TamperState:
    active: set[str] = field(default_factory=set)
    sharpness: float = 0.0
    brightness: float = 0.0
    scene_match: float = 1.0


class TamperDetector:
    def __init__(self, *, persist_s: float = 8.0, blackout_luma: float = 12.0, blur_ratio: float = 0.25,
                 moved_corr: float = 0.35, frozen_s: float = 6.0, learn_s: float = 60.0) -> None:
        self.persist_s = persist_s
        self.blackout_luma = blackout_luma
        self.blur_ratio = blur_ratio
        self.moved_corr = moved_corr
        self.frozen_s = frozen_s
        self.learn_s = learn_s
        self.state = TamperState()
        self._since: dict[str, float] = {}
        self._last_t = -1e9
        # One reference per lighting mode: the same street looks completely different in IR.
        self._sharp_baseline: dict[str, float] = {}
        self._reference: dict[str, np.ndarray] = {}
        self._mode_started: dict[str, float] = {}
        self._last_hash: bytes | None = None
        self._same_since: float | None = None

    def update(self, img: np.ndarray, t: float, mode: str = "day") -> tuple[set[str], set[str]]:
        """Returns (newly raised, newly cleared) conditions."""
        if t - self._last_t < 1.0:
            return set(), set()
        self._last_t = t
        started = self._mode_started.setdefault(mode, t)
        gray = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120).astype(np.float32)
        edges = cv2.resize(edges, (80, 45), interpolation=cv2.INTER_AREA)

        learning = t - started < self.learn_s
        base = self._sharp_baseline.get(mode)
        ref = self._reference.get(mode)
        match = 1.0
        if ref is not None and not learning:
            a = ref - ref.mean()
            b = edges - edges.mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            match = float((a * b).sum() / denom) if denom > 1e-6 else 0.0
        if learning or not self.state.active:
            # fast while learning, then a slow drift so parked cars / seasons don't trip it
            k = 0.1 if learning else 0.01
            self._sharp_baseline[mode] = sharpness if base is None else (1 - k) * base + k * sharpness
            self._reference[mode] = edges if ref is None else (1 - k) * ref + k * edges
        base = self._sharp_baseline.get(mode)

        h = cv2.resize(gray, (32, 18)).tobytes()
        if h == self._last_hash:
            self._same_since = self._same_since if self._same_since is not None else t
        else:
            self._same_since = None
        self._last_hash = h

        now_bad = {
            "blackout": brightness < self.blackout_luma,
            "blur": (not learning and base is not None and base > 20
                     and sharpness < self.blur_ratio * base and brightness >= self.blackout_luma),
            "moved": not learning and match < self.moved_corr and brightness >= self.blackout_luma,
            "frozen": self._same_since is not None and t - self._same_since >= self.frozen_s,
        }
        self.state.sharpness, self.state.brightness, self.state.scene_match = sharpness, brightness, match

        raised, cleared = set(), set()
        for cond, bad in now_bad.items():
            if bad:
                since = self._since.setdefault(cond, t)
                if cond not in self.state.active and (cond == "frozen" or t - since >= self.persist_s):
                    self.state.active.add(cond)
                    raised.add(cond)
            else:
                self._since.pop(cond, None)
                if cond in self.state.active:
                    self.state.active.discard(cond)
                    cleared.add(cond)
        return raised, cleared

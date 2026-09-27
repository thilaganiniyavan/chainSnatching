"""Day / low-light / IR detection and cheap per-frame enhancement.

IR night mode on CCTV cameras produces (near) grayscale frames; colour
cameras at night produce dark, noisy colour frames. The analyser tells them
apart from simple statistics on a small thumbnail, with hysteresis so the
mode doesn't flicker. The active mode picks the rule profile and the
enhancement, and it is reported in every alert.
"""

from __future__ import annotations

import cv2
import numpy as np

from snatchguard.core.types import Lighting


class LightingAnalyzer:
    def __init__(self, *, forced: Lighting | None = None, lowlight_luma: float = 70.0,
                 ir_saturation: float = 12.0, every_s: float = 2.0, confirm: int = 3) -> None:
        self.forced = forced
        self.lowlight_luma = lowlight_luma
        self.ir_saturation = ir_saturation
        self.every_s = every_s
        self.confirm = confirm
        self.mode = forced or Lighting.DAY
        self.luma = 0.0
        self.saturation = 0.0
        self._last_t = -1e9
        self._candidate: Lighting | None = None
        self._votes = 0

    def classify(self, img: np.ndarray) -> Lighting:
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        self.saturation = float(hsv[..., 1].mean())
        self.luma = float(hsv[..., 2].mean())
        b, g, r = (small[..., i].astype(np.int16) for i in range(3))
        channel_spread = float((np.abs(b - g) + np.abs(g - r)).mean())
        if self.saturation < self.ir_saturation and channel_spread < 6.0:
            return Lighting.IR
        if self.luma < self.lowlight_luma:
            return Lighting.LOWLIGHT
        return Lighting.DAY

    def update(self, img: np.ndarray, t: float) -> Lighting:
        if self.forced is not None:
            return self.forced
        if t - self._last_t < self.every_s:
            return self.mode
        self._last_t = t
        guess = self.classify(img)
        if guess == self.mode:
            self._candidate, self._votes = None, 0
        elif guess == self._candidate:
            self._votes += 1
            if self._votes >= self.confirm:
                self.mode, self._candidate, self._votes = guess, None, 0
        else:
            self._candidate, self._votes = guess, 1
        return self.mode


class Enhancer:
    """Contrast/brightness fixes that cost a few ms on a CPU."""

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._gamma_lut = {g: np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)
                           for g in (1.6, 2.0)}

    def apply(self, img: np.ndarray, mode: Lighting) -> np.ndarray:
        if mode is Lighting.DAY:
            return img
        if mode is Lighting.IR:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # colour low-light: brighten, then local contrast on the lightness channel
        luma = float(cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).mean())
        img = cv2.LUT(img, self._gamma_lut[2.0 if luma < 45 else 1.6])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self._clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

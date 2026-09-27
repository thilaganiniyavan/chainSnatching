"""Turn daytime footage into realistic night footage.

Used in two places:
  * the laptop simulator (`simulate_night: ir | lowlight` on a camera) to test
    the whole live pipeline at night without night recordings;
  * training (`training/prepare_dataset.py`) to add night variants of daytime
    images so the detector and pose model learn night appearance.

The effects model what CCTV cameras actually do at night: IR mode gives
grayscale with bright near-field and a dark background (IR fall-off), blown
highlights and sensor noise; colour low-light gives a dark, warm (sodium
street-light) image with heavy noise and motion blur from long exposure.
"""

from __future__ import annotations

import cv2
import numpy as np


def _vignette(h: int, w: int, strength: float) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    cy, cx = h * 0.55, w * 0.5
    d = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
    return np.clip(1.0 - strength * d * 1.6, 0.15, 1.0).astype(np.float32)


def to_ir(img: np.ndarray, rng: np.random.Generator | None = None, strength: float = 1.0) -> np.ndarray:
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.power(gray, 1.0 + 0.6 * strength)                  # darker mid-tones
    falloff = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None]  # IR lights the near field (bottom) most
    gray = gray * (falloff ** strength) * _vignette(h, w, 0.8 * strength)
    gray = np.clip(gray * 1.35, 0, 1)                             # IR bloom on bright surfaces
    gray += rng.normal(0, 0.035 * strength, gray.shape).astype(np.float32)
    out = np.clip(gray * 255, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def to_lowlight(img: np.ndarray, rng: np.random.Generator | None = None, strength: float = 1.0) -> np.ndarray:
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    f = np.power(f, 1.8 + 1.0 * strength) * (0.55 - 0.2 * strength + 0.2)
    f *= np.array([0.65, 0.9, 1.25], np.float32)                 # sodium-lamp colour cast (BGR)
    f *= _vignette(h, w, 0.6 * strength)[..., None]
    f += rng.normal(0, 0.03 * strength, f.shape).astype(np.float32)          # sensor noise
    f += rng.normal(0, 0.02 * strength, (h, w, 1)).astype(np.float32)        # luminance noise
    out = np.clip(f * 255, 0, 255).astype(np.uint8)
    k = int(1 + 2 * strength) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k                                   # slight horizontal motion blur
    return cv2.filter2D(out, -1, kernel)


def make_transform(mode: str | None, seed: int = 0):
    """Frame transform for the simulator: None | 'ir' | 'lowlight'."""
    if not mode or mode == "none":
        return None
    rng = np.random.default_rng(seed)
    fn = {"ir": to_ir, "lowlight": to_lowlight}[mode]
    return lambda img: fn(img, rng)

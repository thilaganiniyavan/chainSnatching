"""Small box / vector helpers. Boxes are numpy arrays (x1, y1, x2, y2)."""

from __future__ import annotations

import math

import numpy as np


def area(b: np.ndarray) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def intersection(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(w * h) if w > 0 and h > 0 else 0.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=float)
    return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()])


def box_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two boxes' closest edges (0 if they overlap)."""
    dx = max(0.0, float(max(a[0], b[0]) - min(a[2], b[2])))
    dy = max(0.0, float(max(a[1], b[1]) - min(a[3], b[3])))
    return math.hypot(dx, dy)


def center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def foot(b: np.ndarray) -> np.ndarray:
    """Bottom-centre: where the object touches the ground."""
    return np.array([(b[0] + b[2]) / 2.0, float(b[3])])


def height(b: np.ndarray) -> float:
    return float(b[3] - b[1])


def width(b: np.ndarray) -> float:
    return float(b[2] - b[0])


def pad_box(b: np.ndarray, frac: float, frame_w: int, frame_h: int) -> np.ndarray:
    w, h = width(b), height(b)
    return np.array([
        max(0.0, b[0] - w * frac),
        max(0.0, b[1] - h * frac),
        min(float(frame_w), b[2] + w * frac),
        min(float(frame_h), b[3] + h * frac),
    ])


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def near_border(b: np.ndarray, frame_w: int, frame_h: int, margin: float) -> bool:
    return b[0] <= margin or b[1] <= margin or b[2] >= frame_w - margin or b[3] >= frame_h - margin

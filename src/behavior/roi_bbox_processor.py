"""Bounding Box Processor — spatial-temporal processing for Interaction ROIs.

Handles:
- Bounding-box interpolation across missing detection gaps
- Exponential Moving Average (EMA) temporal smoothing
- Unphysical outlier rejection based on median box area
- Padding / context expansion with configurable expansion ratios
- Clamping to video frame dimensions
- Strict 1-to-1 indexing alignment between frame numbers and bounding boxes
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ROIBBoxProcessor:
    """Processes, interpolates, smoothes, filters, and expands bounding box sequences.

    Args:
        expansion_ratio: Ratio to expand box width and height (e.g. 0.25 = +25% padding).
        ema_alpha: Alpha parameter for Exponential Moving Average smoothing in (0.0, 1.0].
        max_gap_frames: Maximum gap of missing frames allowed for linear interpolation.
        outlier_std_factor: Multiplier on std-dev from median area for outlier rejection.
        frame_width: Video frame width for clamping (default: 1920).
        frame_height: Video frame height for clamping (default: 1080).
    """

    def __init__(
        self,
        expansion_ratio: float = 0.25,
        ema_alpha: float = 0.4,
        max_gap_frames: int = 15,
        outlier_std_factor: float = 3.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> None:
        self.expansion_ratio = expansion_ratio
        self.ema_alpha = min(max(ema_alpha, 0.01), 1.0)
        self.max_gap_frames = max_gap_frames
        self.outlier_std_factor = outlier_std_factor
        self.frame_width = frame_width
        self.frame_height = frame_height

    def process_sequence(
        self,
        frame_numbers: list[int],
        raw_boxes: list[Optional[tuple[int, int, int, int]]],
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[tuple[int, int, int, int]],
        list[int],
    ]:
        """Process a sequence of raw bounding boxes across a set of frame numbers.

        Args:
            frame_numbers: List of frame indices.
            raw_boxes: Raw bounding boxes [x1, y1, x2, y2] (or None if missing).

        Returns:
            Tuple of:
            - Smoothed bounding boxes sequence
            - Expanded bounding boxes sequence
            - Aligned frame numbers list
        """
        if not frame_numbers or not raw_boxes or len(frame_numbers) != len(raw_boxes):
            return [], [], []

        # 1. Fill missing gaps using linear interpolation
        interpolated = self.interpolate_gaps(frame_numbers, raw_boxes)

        # 2. Reject unphysical box area outliers
        filtered = self.reject_outliers(interpolated)

        # 3. Apply Exponential Moving Average (EMA) temporal smoothing
        smoothed = self.smooth_temporal(filtered)

        # 4. Apply context expansion padding and clamp to frame boundaries
        expanded = [self.expand_bbox(box) for box in smoothed]

        return smoothed, expanded, frame_numbers

    def interpolate_gaps(
        self,
        frames: list[int],
        boxes: list[Optional[tuple[int, int, int, int]]],
    ) -> list[tuple[int, int, int, int]]:
        """Linearly interpolate missing boxes (None) between valid detections."""
        n = len(boxes)
        result: list[Optional[tuple[int, int, int, int]]] = list(boxes)

        # Find indices with valid boxes
        valid_indices = [i for i, b in enumerate(boxes) if b is not None]

        if not valid_indices:
            # Fallback if no valid box exists: return dummy boxes
            return [(0, 0, 100, 100)] * n

        # Fill leading missing frames with first valid box
        first_valid_idx = valid_indices[0]
        for i in range(first_valid_idx):
            result[i] = boxes[first_valid_idx]

        # Fill trailing missing frames with last valid box
        last_valid_idx = valid_indices[-1]
        for i in range(last_valid_idx + 1, n):
            result[i] = boxes[last_valid_idx]

        # Linearly interpolate gaps between valid indices
        for k in range(len(valid_indices) - 1):
            idx1 = valid_indices[k]
            idx2 = valid_indices[k + 1]

            gap_len = idx2 - idx1 - 1
            if 0 < gap_len <= self.max_gap_frames:
                b1 = np.array(boxes[idx1], dtype=float)
                b2 = np.array(boxes[idx2], dtype=float)

                for step in range(1, gap_len + 1):
                    t = step / (gap_len + 1)
                    interp_b = b1 + t * (b2 - b1)
                    result[idx1 + step] = tuple(np.round(interp_b).astype(int))
            elif gap_len > self.max_gap_frames:
                # Forward fill for very long gaps beyond max_gap_frames
                for step in range(1, gap_len + 1):
                    result[idx1 + step] = boxes[idx1]

        # Cast to clean tuple list
        return [tuple(b) if b is not None else (0, 0, 100, 100) for b in result]

    def reject_outliers(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Detect and replace unphysical bounding box size spikes relative to median area."""
        if len(boxes) < 3:
            return list(boxes)

        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=float)
        median_area = np.median(areas)
        std_area = np.std(areas)

        result = list(boxes)

        for i, area in enumerate(areas):
            if std_area > 0 and abs(area - median_area) > self.outlier_std_factor * std_area:
                # Replace outlier with previous valid box
                prev_idx = max(0, i - 1)
                result[i] = result[prev_idx]

        return result

    def smooth_temporal(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Apply Exponential Moving Average (EMA) smoothing across box coordinates."""
        if not boxes:
            return []

        smoothed: list[tuple[int, int, int, int]] = []
        current = np.array(boxes[0], dtype=float)
        smoothed.append(tuple(current.astype(int)))

        for i in range(1, len(boxes)):
            target = np.array(boxes[i], dtype=float)
            current = self.ema_alpha * target + (1.0 - self.ema_alpha) * current
            smoothed.append(tuple(np.round(current).astype(int)))

        return smoothed

    def expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Expand bounding box by expansion_ratio padding and clamp to frame bounds."""
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        pad_x = int(w * self.expansion_ratio / 2.0)
        pad_y = int(h * self.expansion_ratio / 2.0)

        ex_x1 = max(0, x1 - pad_x)
        ex_y1 = max(0, y1 - pad_y)
        ex_x2 = min(self.frame_width, x2 + pad_x)
        ex_y2 = min(self.frame_height, y2 + pad_y)

        return (ex_x1, ex_y1, ex_x2, ex_y2)

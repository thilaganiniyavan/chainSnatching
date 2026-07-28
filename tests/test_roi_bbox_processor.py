"""Unit tests for the ROIBBoxProcessor module.

Tests cover:
- Gap interpolation across missing detection frames
- EMA temporal smoothing
- Unphysical area outlier rejection
- Bounding box context expansion and clamping to frame bounds
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.behavior.roi_bbox_processor import ROIBBoxProcessor


# ======================================================================
# BBox Processor Tests
# ======================================================================

class TestROIBBoxProcessor:

    def test_interpolate_gaps(self):
        processor = ROIBBoxProcessor(max_gap_frames=5)
        frames = [1, 2, 3, 4, 5]
        boxes = [
            (100, 100, 200, 200),
            None,
            None,
            None,
            (200, 200, 300, 300),
        ]

        interp = processor.interpolate_gaps(frames, boxes)
        assert len(interp) == 5
        # Frame 3 (middle gap) should be interpolated midway: (150, 150, 250, 250)
        assert interp[2] == (150, 150, 250, 250)

    def test_smooth_temporal(self):
        processor = ROIBBoxProcessor(ema_alpha=0.5)
        boxes = [
            (100, 100, 200, 200),
            (120, 120, 220, 220),
        ]

        smoothed = processor.smooth_temporal(boxes)
        assert len(smoothed) == 2
        # First box remains un-smoothed
        assert smoothed[0] == (100, 100, 200, 200)
        # Second box EMA: 0.5 * 120 + 0.5 * 100 = 110
        assert smoothed[1] == (110, 110, 210, 210)

    def test_expand_bbox_padding_and_clamping(self):
        processor = ROIBBoxProcessor(
            expansion_ratio=0.5, frame_width=1000, frame_height=1000
        )
        box = (100, 100, 200, 200)  # w=100, h=100 -> pad_x=25, pad_y=25

        expanded = processor.expand_bbox(box)
        assert expanded == (75, 75, 225, 225)

    def test_expand_bbox_clamping_at_edges(self):
        processor = ROIBBoxProcessor(
            expansion_ratio=0.5, frame_width=200, frame_height=200
        )
        box = (10, 10, 190, 190)

        expanded = processor.expand_bbox(box)
        # Should clamp to 0 and max frame dimensions
        assert expanded[0] == 0
        assert expanded[1] == 0
        assert expanded[2] == 200
        assert expanded[3] == 200

    def test_outlier_rejection(self):
        processor = ROIBBoxProcessor(outlier_std_factor=2.0)
        # Constant boxes with one huge unphysical outlier spike
        boxes = [(100, 100, 200, 200)] * 10
        boxes[5] = (0, 0, 1000, 1000)  # Massive outlier

        filtered = processor.reject_outliers(boxes)
        assert filtered[5] != (0, 0, 1000, 1000)
        assert filtered[5] == (100, 100, 200, 200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

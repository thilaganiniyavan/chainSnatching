"""Learned snatch scorer: a pixel-level signal next to the rules (frozen CNN features + calibrated head)."""

from snatchguard.learned.scorer import SnatchScorer, WindowScore, alerts_from_windows, score_video

__all__ = ["SnatchScorer", "WindowScore", "alerts_from_windows", "score_video"]

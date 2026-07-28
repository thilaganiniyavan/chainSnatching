"""Event Visualizer — overlay Behaviour Event information on video frames.

Displays current event label, confidence score, timeline summary,
and interaction duration over the annotated frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.core.models.behaviour_event import BehaviourEvent
from src.core.models.interaction import Interaction
from src.core.models.track import Track


# Event Type -> (B, G, R) Colour Scheme
_EVENT_COLOURS: dict[str, tuple[int, int, int]] = {
    "NORMAL_PASSING": (180, 180, 180),        # Grey
    "VEHICLE_WAITING": (200, 150, 0),         # Cyan-ish
    "FOLLOWING_BEHAVIOUR": (0, 165, 255),      # Orange
    "STATIONARY_INTERACTION": (200, 200, 0),   # Teal
    "CLOSE_ENCOUNTER": (0, 200, 255),         # Yellow
    "SUSPICIOUS_ENCOUNTER": (0, 80, 255),      # Red-Orange
    "RAPID_ESCAPE": (0, 0, 255),              # Red
}


class EventVisualizer:
    """Renders higher-level Behaviour Event HUD overlays on video frames.

    Args:
        font_scale: Font scale for event text.
        panel_alpha: Alpha transparency for background banner.
        fps: Video frame rate for time displays.
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        panel_alpha: float = 0.65,
        fps: float = 30.0,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self.fps = fps if fps > 0 else 30.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        events: list[BehaviourEvent],
        interactions: list[Interaction],
        tracks: list[Track],
    ) -> np.ndarray:
        """Annotate *frame* with Behaviour Event banners and timelines.

        Args:
            frame: Input video frame (BGR).
            events: Detected Behaviour Events for the frame/active interactions.
            interactions: Active interaction objects.
            tracks: Active tracks.

        Returns:
            Annotated frame copy.
        """
        viz = frame.copy()

        track_centers: dict[int, tuple[int, int]] = {
            t.tracking_id: t.center for t in tracks if t.center is not None
        }
        int_map: dict[str, Interaction] = {i.interaction_id: i for i in interactions}

        for event in events:
            interaction = int_map.get(event.interaction_id)
            if not interaction:
                continue

            p_center = track_centers.get(interaction.person_track_id)
            v_center = track_centers.get(interaction.vehicle_track_id)
            if p_center is None or v_center is None:
                continue

            colour = _EVENT_COLOURS.get(event.event_type, (255, 255, 255))
            self._draw_event_banner(viz, event, interaction, p_center, v_center, colour)

        return viz

    def _draw_event_banner(
        self,
        frame: np.ndarray,
        event: BehaviourEvent,
        interaction: Interaction,
        p_center: tuple[int, int],
        v_center: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        """Draw event banner near the interaction midpoint."""
        mid_x = (p_center[0] + v_center[0]) // 2
        mid_y = (p_center[1] + v_center[1]) // 2

        tentative_prefix = "[TENTATIVE] " if event.is_tentative else ""
        lines = [
            f"EVENT: {tentative_prefix}{event.event_type}",
            f"Conf: {event.confidence:.0%} | Dur: {event.duration_frames}f ({event.duration_seconds:.1f}s)",
            f"Seq: {', '.join(event.supporting_sequence[:3])}",
        ]

        # Compute panel size
        padding = 6
        line_height = int(18 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        anchor_x = max(5, mid_x - panel_w // 2)
        anchor_y = max(5, mid_y - panel_h - 20)

        h, w = frame.shape[:2]
        if anchor_x + panel_w > w:
            anchor_x = w - panel_w - 5
        if anchor_y + panel_h > h:
            anchor_y = h - panel_h - 5

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            colour,
            2 if not event.is_tentative else 1,
        )

        # Draw lines
        for i, line in enumerate(lines):
            ty = anchor_y + padding + line_height * (i + 1) - 3
            line_colour = colour if i == 0 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (anchor_x + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )

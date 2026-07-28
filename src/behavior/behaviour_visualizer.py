"""Behaviour Visualizer — overlay behaviour intelligence on video frames.

Renders interaction links, HUD panels, and behavioural state information
using OpenCV drawing primitives.  Designed to be called from the
BehaviourStage and produce a ``behaviour_frame`` for the output video.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.core.models.track import Track


# State → (B, G, R) colour mapping for interaction links
_STATE_COLOURS: dict[InteractionState, tuple[int, int, int]] = {
    InteractionState.NEW: (0, 255, 0),        # Green
    InteractionState.ACTIVE: (255, 180, 0),    # Blue-ish
    InteractionState.LINGERING: (0, 255, 255), # Yellow
    InteractionState.ENDED: (0, 0, 255),       # Red
    InteractionState.ARCHIVED: (128, 128, 128),# Grey
}

# Behaviour type → abbreviated display label
_BEHAVIOUR_LABELS: dict[str, str] = {
    "APPROACHING": "APPROACH",
    "MOVING_AWAY": "MOVING AWAY",
    "FOLLOWING": "FOLLOWING",
    "PARALLEL_MOTION": "PARALLEL",
    "CLOSE_INTERACTION": "CLOSE",
    "STATIONARY_INTERACTION": "STATIONARY",
    "RAPID_ACCELERATION": "RAPID ACCEL",
    "RAPID_DECELERATION": "RAPID DECEL",
    "SUDDEN_DIRECTION_CHANGE": "DIR CHANGE",
    "RAPID_SEPARATION": "RAPID SEP",
    "INTERACTION_DURATION": "DURATION",
}


class BehaviourVisualizer:
    """Renders behaviour annotations on a video frame.

    Args:
        font_scale: Base font scale for labels.
        line_thickness: Thickness of interaction link lines.
        panel_alpha: Alpha for semi-transparent HUD panels (0.0-1.0).
        fps: Video FPS for time display (frames → seconds).
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        line_thickness: int = 2,
        panel_alpha: float = 0.6,
        fps: float = 30.0,
    ) -> None:
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.panel_alpha = panel_alpha
        self.fps = fps if fps > 0 else 30.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        interactions: list[Interaction],
        behaviours: list[BehaviourPrimitive],
        tracks: list[Track],
    ) -> np.ndarray:
        """Produce an annotated copy of *frame* with behaviour overlays.

        Args:
            frame: The base video frame (BGR, uint8).
            interactions: Currently active interactions.
            behaviours: Behaviours detected this frame.
            tracks: All tracked objects this frame.

        Returns:
            A copy of the frame with behaviour visualization overlaid.
        """
        viz = frame.copy()

        # Build fast lookups
        track_centers: dict[int, tuple[int, int]] = {
            t.tracking_id: t.center for t in tracks if t.center is not None
        }

        # Index behaviours by interaction id for quick access
        bhv_index: dict[str, list[BehaviourPrimitive]] = {}
        for bp in behaviours:
            bhv_index.setdefault(bp.interaction_id, []).append(bp)

        for interaction in interactions:
            p_center = track_centers.get(interaction.person_track_id)
            v_center = track_centers.get(interaction.vehicle_track_id)

            if p_center is None or v_center is None:
                continue

            colour = _STATE_COLOURS.get(interaction.state, (255, 255, 255))
            int_behaviours = bhv_index.get(interaction.interaction_id, [])

            # 1. Draw interaction link line
            self._draw_link(viz, p_center, v_center, colour, interaction)

            # 2. Draw HUD panel
            self._draw_hud_panel(
                viz, interaction, int_behaviours, p_center, v_center, colour
            )

        return viz

    # ------------------------------------------------------------------
    # Internal drawing methods
    # ------------------------------------------------------------------

    def _draw_link(
        self,
        frame: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        colour: tuple[int, int, int],
        interaction: Interaction,
    ) -> None:
        """Draw the interaction link line between person and vehicle."""
        cv2.line(frame, pt1, pt2, colour, self.line_thickness)

        # Draw small circles at endpoints
        cv2.circle(frame, pt1, 4, colour, -1)
        cv2.circle(frame, pt2, 4, colour, -1)

        # Interaction ID at midpoint
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(
            frame,
            interaction.interaction_id,
            (mid[0] + 5, mid[1] - 5),
            self._font,
            self.font_scale * 0.9,
            colour,
            1,
        )

    def _draw_hud_panel(
        self,
        frame: np.ndarray,
        interaction: Interaction,
        behaviours: list[BehaviourPrimitive],
        p_center: tuple[int, int],
        v_center: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        """Draw a semi-transparent HUD panel near the interaction midpoint."""

        # Determine the most significant behaviour for display
        primary_behaviour = "—"
        max_confidence = 0.0
        for bp in behaviours:
            if bp.confidence > max_confidence:
                max_confidence = bp.confidence
                primary_behaviour = _BEHAVIOUR_LABELS.get(
                    bp.primitive_type, bp.primitive_type
                )

        # Build text lines
        duration_sec = interaction.duration / self.fps
        lines = [
            f"ID: {interaction.interaction_id}",
            f"State: {interaction.state.value}",
            f"Behaviour: {primary_behaviour}",
            f"Duration: {interaction.duration}f ({duration_sec:.1f}s)",
            f"Distance: {interaction.current_distance:.0f}px",
            f"Rel Speed: {interaction.relative_velocity:+.1f}px/f",
            f"Rel Accel: {interaction.relative_acceleration:+.1f}px/f2",
            f"Confidence: {interaction.interaction_confidence:.0%}",
        ]

        # Position the panel near the vehicle (generally larger bounding box)
        anchor_x = v_center[0] + 15
        anchor_y = v_center[1] - 10

        # Compute panel dimensions
        line_height = int(18 * self.font_scale / 0.45)
        padding = 6
        max_text_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_text_width = max(max_text_width, tw)

        panel_w = max_text_width + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        if anchor_x + panel_w > w:
            anchor_x = max(0, w - panel_w - 5)
        if anchor_y + panel_h > h:
            anchor_y = max(0, h - panel_h - 5)
        if anchor_y < 0:
            anchor_y = 5

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            colour,
            1,
        )

        # Draw text lines
        for i, line in enumerate(lines):
            text_y = anchor_y + padding + line_height * (i + 1) - 3
            text_colour = colour if i < 3 else (200, 200, 200)
            cv2.putText(
                frame,
                line,
                (anchor_x + padding, text_y),
                self._font,
                self.font_scale,
                text_colour,
                1,
            )

"""Behaviour Timeline — chronological event log for interactions.

Maintains a per-interaction timeline of all lifecycle transitions and
behavioural primitives.  This timeline later becomes the explanation layer
used for forensic reports and the Snatch Signature Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive


# ------------------------------------------------------------------
# Domain model
# ------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single entry in a behaviour timeline.

    Attributes:
        frame_number: Video frame at which the event occurred.
        timestamp: Seconds from video start (frame_number / fps).
        event_type: Category identifier such as ``INTERACTION_STARTED`` or
            a behaviour primitive type like ``APPROACHING``.
        interaction_id: The interaction this event belongs to.
        description: Human-readable explanation.
        measurements: Quantitative evidence supporting the event.
    """

    frame_number: int
    timestamp: float
    event_type: str
    interaction_id: str
    description: str = ""
    measurements: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Human-readable descriptions for each event type
# ------------------------------------------------------------------

_EVENT_DESCRIPTIONS: dict[str, str] = {
    # Lifecycle events
    "INTERACTION_STARTED": "Interaction initiated between person and vehicle",
    "INTERACTION_ACTIVE": "Interaction confirmed as active",
    "INTERACTION_LINGERING": "Participants no longer in proximity — interaction lingering",
    "INTERACTION_ENDED": "Interaction has ended",
    # Behaviour primitives
    "APPROACHING": "Vehicle approaching pedestrian",
    "MOVING_AWAY": "Vehicle moving away from pedestrian",
    "FOLLOWING": "Vehicle following pedestrian trajectory",
    "PARALLEL_MOTION": "Parallel motion detected between participants",
    "CLOSE_INTERACTION": "Distance below close-interaction threshold",
    "STATIONARY_INTERACTION": "Both participants near-stationary and close",
    "RAPID_ACCELERATION": "Rapid acceleration detected in interaction",
    "RAPID_DECELERATION": "Rapid deceleration detected in interaction",
    "SUDDEN_DIRECTION_CHANGE": "Sudden heading change between participants",
    "RAPID_SEPARATION": "Rapid separation detected — distance increasing quickly",
    "INTERACTION_DURATION": "Interaction duration milestone reached",
}


# ------------------------------------------------------------------
# Timeline manager
# ------------------------------------------------------------------

class BehaviourTimeline:
    """Manages per-interaction chronological event logs.

    Args:
        fps: Video frame rate used to convert frame numbers to timestamps.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps: float = fps if fps > 0 else 30.0
        self._timelines: dict[str, list[TimelineEvent]] = {}
        # Tracks which lifecycle states we have already recorded per interaction
        self._recorded_states: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        interaction: Interaction,
        behaviours: list[BehaviourPrimitive],
        frame_number: int,
    ) -> None:
        """Record lifecycle and behavioural events for one interaction.

        Should be called once per frame per interaction.
        """
        iid = interaction.interaction_id

        if iid not in self._timelines:
            self._timelines[iid] = []
            self._recorded_states[iid] = set()

        ts = frame_number / self.fps

        # --- Lifecycle events ---
        state_key = interaction.state.value
        if state_key not in self._recorded_states[iid]:
            lifecycle_type = f"INTERACTION_{state_key}"
            if lifecycle_type == "INTERACTION_NEW":
                lifecycle_type = "INTERACTION_STARTED"

            self._timelines[iid].append(
                TimelineEvent(
                    frame_number=frame_number,
                    timestamp=round(ts, 3),
                    event_type=lifecycle_type,
                    interaction_id=iid,
                    description=_EVENT_DESCRIPTIONS.get(
                        lifecycle_type, lifecycle_type
                    ),
                    measurements={
                        "distance": interaction.current_distance,
                        "state": state_key,
                    },
                )
            )
            self._recorded_states[iid].add(state_key)

        # --- Behaviour primitives ---
        for bp in behaviours:
            if bp.interaction_id != iid:
                continue
            self._timelines[iid].append(
                TimelineEvent(
                    frame_number=frame_number,
                    timestamp=round(ts, 3),
                    event_type=bp.primitive_type,
                    interaction_id=iid,
                    description=_EVENT_DESCRIPTIONS.get(
                        bp.primitive_type, bp.primitive_type
                    ),
                    measurements=bp.measurements,
                )
            )

    def get_timeline(self, interaction_id: str) -> list[TimelineEvent]:
        """Return the chronological timeline for a single interaction."""
        return list(self._timelines.get(interaction_id, []))

    def get_all_timelines(self) -> dict[str, list[TimelineEvent]]:
        """Return all recorded timelines keyed by interaction ID."""
        return dict(self._timelines)

    def format_timeline(self, interaction_id: str) -> str:
        """Return a human-readable text representation of an interaction timeline.

        Example output::

            00:03  INTERACTION_STARTED  Interaction initiated between person and vehicle
            00:04  APPROACHING          Vehicle approaching pedestrian
            00:05  CLOSE_INTERACTION    Distance below close-interaction threshold
        """
        events = self._timelines.get(interaction_id, [])
        if not events:
            return f"No timeline recorded for interaction {interaction_id}"

        lines: list[str] = [f"Timeline for {interaction_id}", "=" * 60]
        for evt in events:
            minutes = int(evt.timestamp) // 60
            seconds = evt.timestamp % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            lines.append(f"{time_str}  {evt.event_type:<28s} {evt.description}")
        return "\n".join(lines)

    def format_all_timelines(self) -> str:
        """Format all timelines as a single report string."""
        sections: list[str] = []
        for iid in sorted(self._timelines.keys()):
            sections.append(self.format_timeline(iid))
            sections.append("")
        return "\n".join(sections)

    def to_dict(self, interaction_id: str) -> list[dict]:
        """Serialise a timeline to a list of plain dictionaries."""
        return [
            {
                "frame_number": e.frame_number,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "interaction_id": e.interaction_id,
                "description": e.description,
                "measurements": e.measurements,
            }
            for e in self._timelines.get(interaction_id, [])
        ]

"""Behaviour Logger — structured JSON export for completed interactions.

Generates forensic-grade JSON logs containing the full timeline, behaviour
sequence, motion statistics, and distance statistics for each completed
interaction.  The output schema is designed for downstream consumption by
the Snatch Signature Engine and forensic indexing modules.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.core.models.interaction import Interaction
from src.behavior.behaviour_timeline import TimelineEvent


class BehaviourLogger:
    """Produces structured JSON logs for completed interactions.

    Args:
        fps: Video FPS for converting frame durations to seconds.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps if fps > 0 else 30.0
        self._logs: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        interaction: Interaction,
        timeline: list[TimelineEvent],
    ) -> dict[str, Any]:
        """Build a structured log entry for a single interaction.

        Returns the log dictionary (also stored internally for batch export).
        """
        duration_frames = interaction.duration
        duration_seconds = round(duration_frames / self.fps, 3)

        # Extract the unique ordered behaviour sequence
        behaviour_sequence: list[str] = []
        seen_types: set[str] = set()
        for evt in timeline:
            if (
                not evt.event_type.startswith("INTERACTION_")
                and evt.event_type not in seen_types
            ):
                behaviour_sequence.append(evt.event_type)
                seen_types.add(evt.event_type)

        # Motion statistics
        person_speeds = [
            m.get("person_speed", 0.0) or 0.0
            for m in interaction.motion_history
        ]
        vehicle_speeds = [
            m.get("vehicle_speed", 0.0) or 0.0
            for m in interaction.motion_history
        ]
        distances = [
            r.get("distance", 0.0) for r in interaction.relationship_history
        ]

        log_entry: dict[str, Any] = {
            "interaction_id": interaction.interaction_id,
            "participants": {
                "person_track_id": interaction.person_track_id,
                "vehicle_track_id": interaction.vehicle_track_id,
            },
            "lifecycle": {
                "start_frame": interaction.start_frame,
                "end_frame": interaction.end_frame,
                "duration_frames": duration_frames,
                "duration_seconds": duration_seconds,
                "final_state": interaction.state.value,
            },
            "timeline": [
                {
                    "frame": e.frame_number,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "description": e.description,
                    "measurements": e.measurements,
                }
                for e in timeline
            ],
            "behaviour_sequence": behaviour_sequence,
            "motion_statistics": {
                "person_avg_speed": round(
                    sum(person_speeds) / max(len(person_speeds), 1), 4
                ),
                "person_max_speed": round(max(person_speeds, default=0.0), 4),
                "vehicle_avg_speed": round(
                    sum(vehicle_speeds) / max(len(vehicle_speeds), 1), 4
                ),
                "vehicle_max_speed": round(max(vehicle_speeds, default=0.0), 4),
                "max_relative_velocity": round(
                    max(
                        (abs(interaction.relative_velocity),),
                        default=0.0,
                    ),
                    4,
                ),
            },
            "distance_statistics": {
                "min_distance": round(interaction.min_distance, 4),
                "max_distance": round(interaction.max_distance, 4),
                "avg_distance": round(interaction.avg_distance, 4),
                "final_distance": round(interaction.current_distance, 4),
                "distance_samples": len(distances),
            },
            "behaviour_confidence": round(interaction.interaction_confidence, 4),
        }

        self._logs.append(log_entry)
        return log_entry

    def export_all(self, output_path: str) -> None:
        """Write all logged interactions to a JSON file.

        Args:
            output_path: Absolute path for the output JSON file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"interactions": self._logs, "count": len(self._logs)},
                f,
                indent=2,
                default=str,
            )

    def get_logs(self) -> list[dict[str, Any]]:
        """Return all log entries accumulated so far."""
        return list(self._logs)

    def clear(self) -> None:
        """Reset internal log storage."""
        self._logs.clear()

"""Event Logger — structured JSON and CSV export for Behaviour Events.

Produces ``behaviour_events.json`` and ``behaviour_events.csv`` containing
all detected events, supporting evidence, explanations, and metrics.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.behaviour_event import BehaviourEvent


class EventLogger:
    """Logs and exports Behaviour Events to JSON and CSV formats."""

    def __init__(self) -> None:
        self._events: list[BehaviourEvent] = []

    def log_event(self, event: BehaviourEvent) -> None:
        """Add a BehaviourEvent to internal log storage."""
        self._events.append(event)

    def log_events(self, events: list[BehaviourEvent]) -> None:
        """Add multiple BehaviourEvents to internal log storage."""
        for evt in events:
            self.log_event(evt)

    def export_json(self, output_path: str) -> None:
        """Export all logged events to a structured JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        event_dicts = [self._event_to_dict(e) for e in self._events]

        type_counts: dict[str, int] = {}
        for e in self._events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        payload = {
            "events": event_dicts,
            "summary": {
                "total_events": len(self._events),
                "events_by_type": type_counts,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged events to a tabular CSV file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "event_id",
            "event_type",
            "confidence",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_seconds",
            "person_track_id",
            "vehicle_track_id",
            "interaction_id",
            "supporting_sequence",
            "min_distance_px",
            "peak_relative_velocity",
            "peak_relative_acceleration",
            "is_tentative",
            "explanation",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for e in self._events:
                p_id = e.participants.get("person_track_id", -1)
                v_id = e.participants.get("vehicle_track_id", -1)
                min_dist = e.spatial_evidence.get("min_distance", 0.0)
                peak_vel = e.motion_evidence.get("peak_relative_velocity", 0.0)
                peak_acc = e.motion_evidence.get("peak_relative_acceleration", 0.0)

                writer.writerow(
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type,
                        "confidence": e.confidence,
                        "start_frame": e.start_frame,
                        "end_frame": e.end_frame,
                        "duration_frames": e.duration_frames,
                        "duration_seconds": e.duration_seconds,
                        "person_track_id": p_id,
                        "vehicle_track_id": v_id,
                        "interaction_id": e.interaction_id,
                        "supporting_sequence": " -> ".join(e.supporting_sequence),
                        "min_distance_px": min_dist,
                        "peak_relative_velocity": peak_vel,
                        "peak_relative_acceleration": peak_acc,
                        "is_tentative": e.is_tentative,
                        "explanation": e.explanation,
                    }
                )

    def export_all(self, json_path: str, csv_path: str) -> None:
        """Export both JSON and CSV files."""
        self.export_json(json_path)
        self.export_csv(csv_path)

    def get_events(self) -> list[BehaviourEvent]:
        """Return all logged events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear internal event log storage."""
        self._events.clear()

    @staticmethod
    def _event_to_dict(e: BehaviourEvent) -> dict[str, Any]:
        """Convert a BehaviourEvent to a clean dictionary."""
        return {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "confidence": e.confidence,
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "duration_frames": e.duration_frames,
            "duration_seconds": e.duration_seconds,
            "participants": e.participants,
            "interaction_id": e.interaction_id,
            "supporting_sequence": e.supporting_sequence,
            "motion_evidence": e.motion_evidence,
            "spatial_evidence": e.spatial_evidence,
            "explanation": e.explanation,
            "is_tentative": e.is_tentative,
            "metadata": e.metadata,
        }

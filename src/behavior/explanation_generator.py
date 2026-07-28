"""Explanation Generator — human-readable justification for Behaviour Events.

Generates natural language sentences detailing why a specific Behaviour Event
was classified, referencing duration, spatial boundaries, relative speeds,
accelerations, and the underlying primitive sequence.
"""

from __future__ import annotations

from typing import Any

from src.core.models.interaction import Interaction


class ExplanationGenerator:
    """Generates human-readable explanations from event evidence and metrics."""

    def __init__(self) -> None:
        self._templates: dict[str, str] = {
            "NORMAL_PASSING": (
                "{vehicle_label} approached {person_label} for {duration:.1f} seconds, "
                "reached a minimum distance of {min_dist:.0f} pixels, and departed "
                "without close interaction."
            ),
            "VEHICLE_WAITING": (
                "{vehicle_label} remained idle near {person_label} for {duration:.1f} seconds "
                "at an average distance of {avg_dist:.0f} pixels."
            ),
            "FOLLOWING_BEHAVIOUR": (
                "{vehicle_label} followed {person_label} trajectory for {duration:.1f} seconds "
                "with an average distance of {avg_dist:.0f} pixels."
            ),
            "STATIONARY_INTERACTION": (
                "{vehicle_label} and {person_label} remained stationary together for {duration:.1f} seconds "
                "within {min_dist:.0f} pixels."
            ),
            "CLOSE_ENCOUNTER": (
                "{vehicle_label} encountered {person_label} for {duration:.1f} seconds, "
                "closing to a minimum distance of {min_dist:.0f} pixels."
            ),
            "SUSPICIOUS_ENCOUNTER": (
                "{vehicle_label} approached {person_label} for {duration:.1f} seconds, "
                "remained within {min_dist:.0f} pixels, accelerated by {peak_accel:.1f} px/frame², "
                "and rapidly separated."
            ),
            "RAPID_ESCAPE": (
                "{vehicle_label} rapidly separated from {person_label} after {duration:.1f} seconds, "
                "accelerating by {peak_accel:.1f} px/frame² up to relative speed {peak_vel:.1f} px/frame."
            ),
        }

    def register_template(self, event_type: str, template: str) -> None:
        """Register or override an explanation template string."""
        self._templates[event_type] = template

    def generate(
        self,
        event_type: str,
        interaction: Interaction,
        motion_evidence: dict[str, Any],
        spatial_evidence: dict[str, Any],
        supporting_sequence: list[str],
        fps: float = 30.0,
    ) -> str:
        """Generate a natural language explanation string for an event."""

        duration_sec = interaction.duration / (fps if fps > 0 else 30.0)
        person_label = f"pedestrian (Track {interaction.person_track_id})"
        vehicle_label = f"vehicle (Track {interaction.vehicle_track_id})"

        min_dist = spatial_evidence.get("min_distance", interaction.min_distance)
        avg_dist = spatial_evidence.get("avg_distance", interaction.avg_distance)
        final_dist = spatial_evidence.get("final_distance", interaction.current_distance)

        peak_vel = motion_evidence.get("peak_relative_velocity", 0.0)
        peak_accel = motion_evidence.get("peak_relative_acceleration", 0.0)

        template = self._templates.get(event_type)

        if template:
            try:
                return template.format(
                    person_label=person_label,
                    vehicle_label=vehicle_label,
                    duration=duration_sec,
                    min_dist=min_dist,
                    avg_dist=avg_dist,
                    final_dist=final_dist,
                    peak_vel=peak_vel,
                    peak_accel=peak_accel,
                )
            except KeyError:
                pass

        # Default fallback explanation
        seq_str = ", ".join(supporting_sequence) if supporting_sequence else "none"
        return (
            f"{event_type} event involving {vehicle_label} and {person_label} "
            f"lasting {duration_sec:.1f}s (min dist: {min_dist:.0f}px, sequence: [{seq_str}])."
        )

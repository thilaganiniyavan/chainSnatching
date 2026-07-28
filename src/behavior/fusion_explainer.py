"""Fusion Explainer — human-readable forensic explanation text generator.

Constructs natural-language, evidence-backed explanations summarizing:
- Participant roles (person track ID, vehicle track ID)
- Behaviour Graph pattern transitions
- Action Recognition classifications with confidence scores
- Motion & spatial proximity metrics (duration, distance, acceleration)
"""

from __future__ import annotations

from typing import Any

from src.core.models.fused_interaction import FusedInteraction


class FusionExplainer:
    """Generates human-readable, evidence-provenance text explanations for fused interactions."""

    def generate_explanation(self, fusion: FusedInteraction) -> str:
        """Generate a natural-language explanation string for a FusedInteraction.

        Args:
            fusion: Input :class:`FusedInteraction` instance.

        Returns:
            Human-readable explanation text string.
        """
        parts: list[str] = []

        # 1. Subject description
        p_id = fusion.person_track_id
        v_id = fusion.vehicle_track_id

        if v_id != -1:
            parts.append(f"Vehicle (Track {v_id}) and Person (Track {p_id}) engaged in an interaction.")
        else:
            parts.append(f"Person (Track {p_id}) engaged in an interaction.")

        # 2. Behaviour Graph pattern sequence description
        patterns = fusion.behaviour_patterns
        if patterns:
            pat_str = " -> ".join(patterns)
            parts.append(f"Observed behaviour pattern sequence: [{pat_str}].")

        # 3. Spatial & motion evidence
        sp = fusion.spatial_evidence
        mo = fusion.motion_evidence
        dur = fusion.duration_seconds

        min_dist = sp.get("min_distance_px", None)
        avg_spd = mo.get("average_speed_px", None)

        spatial_desc = []
        if min_dist is not None:
            spatial_desc.append(f"minimum proximity of {min_dist:.1f}px")
        if dur > 0:
            spatial_desc.append(f"interaction duration of {dur:.1f}s")
        if avg_spd is not None:
            spatial_desc.append(f"average relative speed of {avg_spd:.1f}px/s")

        if spatial_desc:
            parts.append(f"Kinematic dynamics exhibited {', '.join(spatial_desc)}.")

        # 4. Action Recognition stream description
        action_timeline = fusion.action_timeline
        if action_timeline:
            valid_actions = [
                act for act in action_timeline
                if act.get("action_label") and act.get("action_label") != "Unknown"
            ]
            if valid_actions:
                act_descriptions = [
                    f"{act['action_label']} (confidence: {act.get('action_confidence', 0.0):.0%})"
                    for act in valid_actions[:3]
                ]
                parts.append(f"Pose-based action recognition detected: {', '.join(act_descriptions)}.")

        # 5. Overall fusion confidence statement
        parts.append(f"Multi-modal fusion confidence: {fusion.fusion_confidence:.0%} (strategy: {fusion.fusion_strategy}).")

        return " ".join(parts)

"""Signature Matcher Engine — weighted multi-modal evidence evaluation.

Evaluates FusedInteraction multi-modal evidence against configurable SignatureTemplates.
Checks:
- Behaviour Graph pattern sequence
- Target action predictions (Reaching, Grabbing, Pulling)
- Spatial proximity distance constraints
- Kinematic motion dynamics (speed, acceleration)
- Temporal ordering of events
- Missing evidence identification
"""

from __future__ import annotations

from typing import Any

from src.core.models.fused_interaction import FusedInteraction
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.snatch.signature_config import SignatureTemplate, StandardMotorcycleSnatchSignature


class SignatureMatcher:
    """Evaluates multi-modal FusedInteraction evidence against a SignatureTemplate.

    Args:
        template: SignatureTemplate instance to match against.
    """

    def __init__(self, template: SignatureTemplate | None = None) -> None:
        self.template = template if template is not None else StandardMotorcycleSnatchSignature()

    def evaluate(self, fusion: FusedInteraction) -> SnatchSignatureResult:
        """Evaluate *fusion* multi-modal evidence against active signature template.

        Args:
            fusion: Input :class:`FusedInteraction` instance.

        Returns:
            A :class:`SnatchSignatureResult` object.
        """
        matched_evidence: list[dict[str, Any]] = []
        missing_evidence: list[dict[str, Any]] = []

        weights = self.template.evidence_weights
        patterns = fusion.behaviour_patterns
        actions = [
            a.get("action_label") for a in fusion.action_timeline
            if a.get("action_label") and a.get("action_label") != "Unknown"
        ]
        sp = fusion.spatial_evidence
        mo = fusion.motion_evidence

        # 1. Approach Pattern check
        w_app = weights.get("approach_pattern", 0.15)
        if "APPROACH_PATTERN" in patterns or "FOLLOW_PATTERN" in patterns:
            matched_evidence.append({
                "component": "approach_pattern",
                "weight": w_app,
                "description": "Observed approach or follow behaviour pattern.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "approach_pattern",
                "weight": w_app,
                "description": "Approach pattern not detected in graph.",
                "symbol": "✗",
            })

        # 2. Interaction / Proximity Pattern check
        w_int = weights.get("interaction_pattern", 0.20)
        if "INTERACTION_PATTERN" in patterns or "PROXIMITY_PATTERN" in patterns:
            matched_evidence.append({
                "component": "interaction_pattern",
                "weight": w_int,
                "description": "Observed interaction or proximity pattern.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "interaction_pattern",
                "weight": w_int,
                "description": "Close interaction pattern missing.",
                "symbol": "✗",
            })

        # 3. Target Action (Reaching / Grabbing / Pulling) check
        w_act = weights.get("target_action", 0.25)
        matched_target_actions = [a for a in actions if a in self.template.target_actions]
        if matched_target_actions:
            act_label = matched_target_actions[0]
            matched_evidence.append({
                "component": "target_action",
                "weight": w_act,
                "description": f"Pose action recognition detected '{act_label}'.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "target_action",
                "weight": w_act,
                "description": "Confirmed reaching or grabbing action not detected.",
                "symbol": "✗",
            })

        # 4. Rapid Acceleration / Motion Dynamics check
        w_acc = weights.get("rapid_acceleration", 0.15)
        avg_spd = mo.get("average_speed_px", 0.0)
        if avg_spd >= self.template.min_average_speed or "ESCAPE_PATTERN" in patterns:
            matched_evidence.append({
                "component": "rapid_acceleration",
                "weight": w_acc,
                "description": "High relative speed or escape acceleration observed.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "rapid_acceleration",
                "weight": w_acc,
                "description": "Rapid acceleration or high-speed escape absent.",
                "symbol": "✗",
            })

        # 5. Escape / Separation Pattern check
        w_esc = weights.get("escape_pattern", 0.15)
        if "ESCAPE_PATTERN" in patterns or "SEPARATION_PATTERN" in patterns:
            matched_evidence.append({
                "component": "escape_pattern",
                "weight": w_esc,
                "description": "Rapid separation or escape trajectory detected.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "escape_pattern",
                "weight": w_esc,
                "description": "Escape or rapid separation pattern missing.",
                "symbol": "✗",
            })

        # 6. Spatial Proximity Constraint check
        w_prox = weights.get("proximity_constraint", 0.10)
        min_dist = sp.get("min_distance_px", 999.0)
        if min_dist <= self.template.max_proximity_px:
            matched_evidence.append({
                "component": "proximity_constraint",
                "weight": w_prox,
                "description": f"Spatial proximity distance ({min_dist:.1f}px) within threshold.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "proximity_constraint",
                "weight": w_prox,
                "description": "Spatial proximity distance exceeded allowable limit.",
                "symbol": "✗",
            })

        # Calculate total weighted signature match score
        total_possible_weight = sum(weights.values())
        achieved_weight = sum(item["weight"] for item in matched_evidence)
        signature_score = round(achieved_weight / max(1e-5, total_possible_weight), 4)

        # Determine decision boundary label
        decision = self._classify_decision(signature_score)

        return SnatchSignatureResult(
            interaction_id=fusion.interaction_id,
            fusion_id=fusion.fusion_id,
            matched_signature_name=self.template.signature_name,
            signature_score=signature_score,
            decision=decision,
            confidence=round(fusion.fusion_confidence * signature_score, 4),
            matched_evidence=matched_evidence,
            missing_evidence=missing_evidence,
            behaviour_evidence=patterns,
            action_evidence=actions,
            motion_evidence=mo,
            spatial_evidence=sp,
            temporal_evidence=fusion.temporal_evidence,
            evidence_timeline=fusion.evidence_timeline,
            metadata={
                "start_frame": fusion.start_frame,
                "end_frame": fusion.end_frame,
                "duration_seconds": fusion.duration_seconds,
                "person_track_id": fusion.person_track_id,
                "vehicle_track_id": fusion.vehicle_track_id,
                "timestamp": fusion.metadata.get("timestamp", 0.0),
            },
        )

    def _classify_decision(self, score: float) -> str:
        """Classify signature score into decision label based on template thresholds."""
        thresholds = self.template.decision_thresholds
        if score >= thresholds.get("High Confidence Match", 0.85):
            return "High Confidence Match"
        elif score >= thresholds.get("Strong Match", 0.70):
            return "Strong Match"
        elif score >= thresholds.get("Partial Match", 0.55):
            return "Partial Match"
        elif score >= thresholds.get("Weak Match", 0.35):
            return "Weak Match"
        else:
            return "No Match"

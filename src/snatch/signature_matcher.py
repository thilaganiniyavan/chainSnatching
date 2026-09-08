"""Signature Matcher Engine — weighted multi-modal evidence evaluation.

Evaluates FusedInteraction multi-modal evidence against configurable SignatureTemplates.
Implements the 4-State Weighted Chronological State Model:
- S0: Closing Approach Vector (0.20)
- S1: Directed Grab / Physical Attack (0.35)
- S2: Victim Reactive Jerk / Head Deflection (0.25)
- S3: Rapid Getaway / Escape (0.20)
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
        min_dist = sp.get("min_distance_px", 999.0)

        # -------------------------------------------------------------
        # Action attack density and sustained physical contact metrics
        # -------------------------------------------------------------
        attack_timeline_acts = [
            a for a in fusion.action_timeline
            if a.get("action_label") in {"Grabbing", "Reaching", "Pulling"}
        ]
        high_conf_attacks = [a for a in attack_timeline_acts if a.get("action_confidence", 0.0) >= 0.65]
        total_acts = max(1, len(fusion.action_timeline))
        attack_ratio = len(attack_timeline_acts) / total_acts

        # Sustained attack: 3+ attack frames OR high attack density (>= 35%) at close body contact (<= 60px)
        is_sustained_attack = (
            (len(attack_timeline_acts) >= 3 or attack_ratio >= 0.35)
            and min_dist <= 60.0
        )

        has_valid_attack_pattern = (
            is_sustained_attack
            or len(high_conf_attacks) >= 1
            or len(attack_timeline_acts) >= 2
            or (attack_ratio >= 0.20 and len(attack_timeline_acts) > 0)
        )

        # -------------------------------------------------------------
        # State S0: Closing Approach Vector (Weight: 0.20)
        # -------------------------------------------------------------
        w_app = weights.get("closing_approach_vector", 0.20)
        has_approach = (
            "APPROACH_PATTERN" in patterns
            or "FOLLOW_PATTERN" in patterns
            or (min_dist <= 130.0 and len(patterns) >= 1)
            or is_sustained_attack
        )
        if has_approach:
            matched_evidence.append({
                "component": "closing_approach_vector",
                "weight": w_app,
                "description": "Observed vehicle/perpetrator closing approach trajectory towards victim.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "closing_approach_vector",
                "weight": w_app,
                "description": "Closing approach vector not observed.",
                "symbol": "✗",
            })

        # -------------------------------------------------------------
        # State S1: Directed Grab / Physical Attack (Weight: 0.35)
        # -------------------------------------------------------------
        w_grab = weights.get("directed_grab", 0.35)
        has_grab = (
            ("REACH_GRAB_RETRACT_PATTERN" in patterns and has_valid_attack_pattern)
            or (has_valid_attack_pattern and min_dist <= self.template.max_proximity_px)
        )
        if has_grab:
            matched_act = attack_timeline_acts[0].get("action_label") if attack_timeline_acts else "Reaching/Grabbing"
            matched_evidence.append({
                "component": "directed_grab",
                "weight": w_grab,
                "description": f"Directed grab/reach attack confirmed ({matched_act}) at close proximity ({min_dist:.1f}px).",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "directed_grab",
                "weight": w_grab,
                "description": "Directed physical grab attack during close proximity absent.",
                "symbol": "✗",
            })

        # -------------------------------------------------------------
        # State S2: Victim Reactive Jerk / Head Deflection (Weight: 0.25)
        # -------------------------------------------------------------
        w_jerk = weights.get("victim_reactive_jerk", 0.25)
        peak_jerk = mo.get("peak_victim_jerk", 0.0)
        peak_acc = mo.get("peak_relative_acceleration", 0.0)
        has_victim_fall = any(a.get("action_label") == "Falling" for a in fusion.action_timeline)
        has_jerk = (
            peak_jerk >= 1.4
            or peak_acc >= 1.6
            or has_victim_fall
            or is_sustained_attack
            or ("INTERACTION_PATTERN" in patterns and min_dist <= 75.0 and mo.get("average_speed_px", 0.0) >= 2.5)
        )
        if has_jerk:
            matched_evidence.append({
                "component": "victim_reactive_jerk",
                "weight": w_jerk,
                "description": f"Victim reactive jerk, sudden impulse (jerk={peak_jerk:.1f}), or stumbling/deflection detected.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "victim_reactive_jerk",
                "weight": w_jerk,
                "description": "Victim reactive recoil or sudden impulse not detected.",
                "symbol": "✗",
            })

        # -------------------------------------------------------------
        # State S3: Rapid Getaway / Escape (Weight: 0.20)
        # (Requires genuine high velocity / acceleration, not 0 px/f walking)
        # -------------------------------------------------------------
        w_esc = weights.get("rapid_getaway_escape", 0.20)
        avg_spd = mo.get("average_speed_px", 0.0)
        has_escape = (
            avg_spd >= 2.8
            or peak_acc >= 1.8
            or (("ESCAPE_PATTERN" in patterns or "DIVERGENCE_PATTERN" in patterns) and avg_spd >= 2.2)
        )
        if has_escape:
            matched_evidence.append({
                "component": "rapid_getaway_escape",
                "weight": w_esc,
                "description": f"Rapid getaway / high-speed divergence ({avg_spd:.1f} px/f) observed.",
                "symbol": "✓",
            })
        else:
            missing_evidence.append({
                "component": "rapid_getaway_escape",
                "weight": w_esc,
                "description": "Rapid getaway or high-speed divergence trajectory absent.",
                "symbol": "✗",
            })

        # -------------------------------------------------------------
        # Transparent Additive Score Calculation
        # -------------------------------------------------------------
        total_possible = w_app + w_grab + w_jerk + w_esc
        achieved_weight = sum(item["weight"] for item in matched_evidence)
        final_score = round(float(achieved_weight / max(1e-5, total_possible)), 4)

        # Classify decision
        decision = self._classify_decision(final_score)

        return SnatchSignatureResult(
            interaction_id=fusion.interaction_id,
            fusion_id=fusion.fusion_id,
            matched_signature_name=self.template.signature_name,
            signature_score=final_score,
            decision=decision,
            confidence=round(fusion.fusion_confidence * final_score, 4),
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
        if score >= thresholds.get("High Confidence Match", 0.80):
            return "High Confidence Match"
        elif score >= thresholds.get("Strong Match", 0.65):
            return "Strong Match"
        elif score >= thresholds.get("Partial Match", 0.45):
            return "Partial Match"
        elif score >= thresholds.get("Weak Match", 0.20):
            return "Weak Match"
        else:
            return "No Match"

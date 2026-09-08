"""Behaviour Fusion Engine — multi-modal evidence fusion manager.

Combines Stream A (Behaviour Graph patterns & spatial/motion statistics) and
Stream B (Human Action Recognition classifications & ST-GCN confidence scores) into
a unified, explainable FusedInteraction domain model.

Exposes clean API suite:
- fuse_interaction()
- update_fusion()
- finalize_fusion()
- get_fused_interaction()
- get_completed_fusions()
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.core.models.behaviour_graph import BehaviourGraph
from src.core.models.action_result import ActionResult
from src.core.models.fused_interaction import FusedInteraction
from src.core.models.track import Track
from src.behavior.fusion_temporal_aligner import FusionTemporalAligner
from src.behavior.fusion_strategies import FusionStrategyEngine
from src.behavior.fusion_explainer import FusionExplainer


class BehaviourFusionEngine:
    """Engine executing multi-modal evidence fusion between Behaviour Graphs and Action Recognition.

    Args:
        fusion_strategy: Strategy name ("weighted_confidence", "bayesian", "rule_based", "voting_based", "weighted_averaging").
        fps: Video frame rate.
    """

    def __init__(
        self,
        fusion_strategy: str = "weighted_confidence",
        fps: float = 30.0,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.strategy_engine = FusionStrategyEngine(strategy=fusion_strategy)
        self.aligner = FusionTemporalAligner()
        self.explainer = FusionExplainer()

        # Storage: fusion_id -> FusedInteraction
        self._fused_map: dict[str, FusedInteraction] = {}
        # Accumulated action results per interaction: interaction_id -> list[ActionResult]
        self._action_history: dict[str, list[ActionResult]] = {}
        # Historical minimum spatial distance per interaction
        self._min_dist_history: dict[str, float] = {}

    def fuse_interaction(
        self,
        graph: BehaviourGraph,
        action_results: list[ActionResult],
        tracks: list[Track] | None = None,
    ) -> FusedInteraction:
        """Create and fuse a new FusedInteraction from BehaviourGraph and ActionResults."""
        iid = graph.interaction_id
        fusion_id = f"FUSED-{iid}"

        # Filter action results for this interaction
        current_frame_actions = [
            act for act in action_results if act.interaction_id == iid or act.sequence_id.startswith(f"SEQ-{iid}")
        ]

        # Accumulate action results across interaction lifetime so transient actions (grab, pull) are preserved
        if iid not in self._action_history:
            self._action_history[iid] = []

        seen_action_ids = {a.action_id for a in self._action_history[iid]}
        for act in current_frame_actions:
            if act.action_id not in seen_action_ids:
                self._action_history[iid].append(act)
                seen_action_ids.add(act.action_id)

        all_actions = self._action_history[iid] if self._action_history[iid] else current_frame_actions

        # Extract patterns
        patterns = [n.pattern_type for n in graph.nodes]

        # Compute confidence scores via Strategy Engine
        b_conf, a_conf, f_conf = self.strategy_engine.fuse(
            graph=graph,
            action_results=all_actions,
            motion_conf=0.80,
        )

        # Align multi-modal evidence timeline
        evidence_timeline = self.aligner.align_streams(
            graph=graph,
            action_results=all_actions,
            fps=self.fps,
        )

        # Extract motion & spatial evidence
        motion_evidence = self._extract_motion_evidence(graph, tracks)
        spatial_evidence = self._extract_spatial_evidence(graph, tracks)

        # Multi-modal synthesis: If ST-GCN action detected physical attack (Grabbing, Reaching, Pulling)
        # during close spatial proximity (<= 140px), synthesize REACH_GRAB_RETRACT_PATTERN
        min_d = spatial_evidence.get("min_distance_px", 999.0)
        attack_acts = [
            act for act in all_actions
            if act.predicted_action in {"Grabbing", "Reaching", "Pulling"}
        ]
        high_conf_attacks = [a for a in attack_acts if a.action_confidence >= 0.65]
        total_acts = max(1, len(all_actions))
        attack_ratio = len(attack_acts) / total_acts

        # Require high confidence (>= 0.65) or multiple attack frames (not a 1-frame outlier in a sea of walking)
        has_attack_act = (
            (len(high_conf_attacks) >= 1 or len(attack_acts) >= 2 or attack_ratio >= 0.20)
            and len(attack_acts) > 0
        )

        if has_attack_act and min_d <= 140.0:
            if "REACH_GRAB_RETRACT_PATTERN" not in patterns:
                patterns.append("REACH_GRAB_RETRACT_PATTERN")
            motion_evidence["is_reach_retract_impulse"] = True

        start_f = graph.start_frame if graph.start_frame is not None else 0
        end_f = (
            graph.end_frame
            if (graph.end_frame is not None and graph.end_frame > 0)
            else (evidence_timeline[-1]["frame"] if evidence_timeline else start_f)
        )
        frame_cnt = max(1, end_f - start_f + 1)
        duration_sec = round(frame_cnt / self.fps, 3)

        action_timeline = [
            {
                "frame": act.metadata.get("frame_index", start_f),
                "action_label": act.predicted_action,
                "action_confidence": act.action_confidence,
                "model_name": act.model_name,
            }
            for act in all_actions
        ]

        fused = FusedInteraction(
            fusion_id=fusion_id,
            interaction_id=iid,
            person_track_id=graph.person_track_id,
            vehicle_track_id=graph.vehicle_track_id,
            start_frame=start_f,
            end_frame=end_f,
            duration_seconds=duration_sec,
            behaviour_patterns=patterns,
            action_timeline=action_timeline,
            motion_evidence=motion_evidence,
            spatial_evidence=spatial_evidence,
            temporal_evidence={"start_frame": start_f, "end_frame": end_f, "duration_seconds": duration_sec},
            action_evidence=[
                {"action_id": act.action_id, "label": act.predicted_action, "confidence": act.action_confidence}
                for act in all_actions
            ],
            behaviour_confidence=b_conf,
            action_confidence=a_conf,
            fusion_confidence=f_conf,
            fusion_strategy=self.strategy_engine.strategy,
            evidence_timeline=evidence_timeline,
        )

        # Generate human-readable explanation text
        fused.explanation_text = self.explainer.generate_explanation(fused)

        self._fused_map[fusion_id] = fused
        return fused

    def update_fusion(
        self,
        fusion_id: str,
        graph: BehaviourGraph,
        action_results: list[ActionResult],
    ) -> FusedInteraction:
        """Update an active FusedInteraction with new frame evidence."""
        return self.fuse_interaction(graph, action_results)

    def finalize_fusion(self, fusion_id: str) -> Optional[FusedInteraction]:
        """Finalize a FusedInteraction."""
        return self._fused_map.get(fusion_id)

    def get_fused_interaction(self, fusion_id: str) -> Optional[FusedInteraction]:
        """Return FusedInteraction by fusion_id."""
        return self._fused_map.get(fusion_id)

    def get_completed_fusions(self) -> list[FusedInteraction]:
        """Return all completed FusedInteraction objects."""
        return list(self._fused_map.values())

    @staticmethod
    def _extract_motion_evidence(
        graph: BehaviourGraph, tracks: list[Track] | None
    ) -> dict[str, Any]:
        """Extract motion statistics dictionary."""
        evidence: dict[str, Any] = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        }
        if tracks:
            track_map = {t.tracking_id: t for t in tracks}
            p_tr = track_map.get(graph.person_track_id)
            v_tr = track_map.get(graph.vehicle_track_id)
            speeds = [t.average_speed for t in (p_tr, v_tr) if t and t.average_speed is not None]
            if speeds:
                evidence["average_speed_px"] = round(max(speeds), 2)
            if p_tr and p_tr.direction is not None:
                evidence["direction_deg"] = round(p_tr.direction, 1)
        return evidence

    def _extract_spatial_evidence(
        self,
        graph: BehaviourGraph,
        tracks: list[Track] | None,
    ) -> dict[str, Any]:
        """Extract spatial relationship statistics dictionary."""
        evidence: dict[str, Any] = {
            "has_vehicle": graph.vehicle_track_id != -1,
        }
        if tracks:
            t1 = next((t for t in tracks if t.tracking_id == graph.person_track_id), None)
            t2 = next((t for t in tracks if t.tracking_id == graph.vehicle_track_id), None)
            if t1 and t2:
                current_dist = float(np.linalg.norm(np.array(t1.center) - np.array(t2.center)))
                prev_min = self._min_dist_history.get(graph.interaction_id, current_dist)
                min_d = min(prev_min, current_dist)
                self._min_dist_history[graph.interaction_id] = min_d
                evidence["min_distance_px"] = round(min_d, 1)
            elif graph.interaction_id in self._min_dist_history:
                evidence["min_distance_px"] = round(self._min_dist_history[graph.interaction_id], 1)
        return evidence

    def clear(self) -> None:
        """Clear internal storage."""
        self._fused_map.clear()

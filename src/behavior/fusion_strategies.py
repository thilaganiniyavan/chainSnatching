"""Fusion Strategies Engine — multi-modal confidence fusion algorithms.

Supports 5 configurable evidence fusion strategies:
1. ``weighted_confidence``: Weighted combination of graph, action, and motion confidences.
2. ``bayesian``: Bayesian posterior probability joint evidence update.
3. ``rule_based``: Multi-modal pattern-action rule logic fusion.
4. ``voting_based``: Consensus voting across evidence streams.
5. ``weighted_averaging``: Standard weighted mean averaging.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.models.behaviour_graph import BehaviourGraph
from src.core.models.action_result import ActionResult


class FusionStrategyEngine:
    """Computes overall fusion confidence score and evidence weights across streams.

    Args:
        strategy: Fusion strategy name ("weighted_confidence", "bayesian", "rule_based", "voting_based", "weighted_averaging").
        w_graph: Weight for Behaviour Graph evidence stream (default: 0.40).
        w_action: Weight for Action Recognition evidence stream (default: 0.40).
        w_motion: Weight for Motion Trajectory evidence stream (default: 0.20).
    """

    def __init__(
        self,
        strategy: str = "weighted_confidence",
        w_graph: float = 0.40,
        w_action: float = 0.40,
        w_motion: float = 0.20,
    ) -> None:
        self.strategy = strategy.lower().strip()
        self.w_graph = w_graph
        self.w_action = w_action
        self.w_motion = w_motion

        # Normalize weights to sum to 1.0
        total_w = max(1e-5, self.w_graph + self.w_action + self.w_motion)
        self.w_graph /= total_w
        self.w_action /= total_w
        self.w_motion /= total_w

    def fuse(
        self,
        graph: BehaviourGraph,
        action_results: list[ActionResult],
        motion_conf: float = 0.80,
    ) -> tuple[float, float, float]:
        """Compute (behaviour_confidence, action_confidence, fusion_confidence).

        Args:
            graph: Source BehaviourGraph object.
            action_results: List of ActionResult objects.
            motion_conf: Motion trajectory evidence confidence.

        Returns:
            Tuple of ``(behaviour_confidence, action_confidence, fusion_confidence)``.
        """
        # 1. Behaviour Graph stream confidence
        graph_confs = [node.confidence for node in graph.nodes]
        behaviour_conf = float(np.mean(graph_confs)) if graph_confs else 0.50

        # 2. Action Recognition stream confidence
        action_confs = [act.action_confidence for act in action_results if act.predicted_action != "Unknown"]
        action_conf = float(np.mean(action_confs)) if action_confs else 0.50

        if self.strategy == "bayesian":
            fusion_conf = self._bayesian_fusion(behaviour_conf, action_conf, motion_conf)
        elif self.strategy == "rule_based":
            fusion_conf = self._rule_based_fusion(graph, action_results, behaviour_conf, action_conf)
        elif self.strategy == "voting_based":
            fusion_conf = self._voting_fusion(behaviour_conf, action_conf, motion_conf)
        elif self.strategy == "weighted_averaging":
            fusion_conf = (behaviour_conf + action_conf + motion_conf) / 3.0
        else: # Default: "weighted_confidence"
            fusion_conf = (
                self.w_graph * behaviour_conf
                + self.w_action * action_conf
                + self.w_motion * motion_conf
            )

        return (
            round(behaviour_conf, 4),
            round(action_conf, 4),
            round(float(min(1.0, max(0.0, fusion_conf))), 4),
        )

    def _bayesian_fusion(
        self,
        p_graph: float,
        p_action: float,
        p_motion: float,
    ) -> float:
        """Bayesian posterior joint probability update assuming independent evidence streams."""
        eps = 1e-4
        p1 = min(max(p_graph, eps), 1.0 - eps)
        p2 = min(max(p_action, eps), 1.0 - eps)
        p3 = min(max(p_motion, eps), 1.0 - eps)

        num = p1 * p2 * p3
        den = num + (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
        return float(num / max(eps, den))

    def _rule_based_fusion(
        self,
        graph: BehaviourGraph,
        action_results: list[ActionResult],
        b_conf: float,
        a_conf: float,
    ) -> float:
        """Rule-based fusion logic combining specific pattern-action co-occurrences."""
        patterns = {n.pattern_type for n in graph.nodes}
        actions = {act.predicted_action for act in action_results}

        bonus = 0.0
        # Rule 1: High interaction pattern co-occurring with Reaching / Grabbing
        if "INTERACTION_PATTERN" in patterns and ("Reaching" in actions or "Grabbing" in actions):
            bonus += 0.15

        # Rule 2: Approach or Follow co-occurring with Approaching / Running
        if ("APPROACH_PATTERN" in patterns or "FOLLOW_PATTERN" in patterns) and ("Approaching" in actions or "Running" in actions):
            bonus += 0.10

        # Rule 3: Escape co-occurring with Running
        if "ESCAPE_PATTERN" in patterns and "Running" in actions:
            bonus += 0.15

        base_conf = 0.5 * b_conf + 0.5 * a_conf
        return float(base_conf + bonus)

    def _voting_fusion(
        self,
        p_graph: float,
        p_action: float,
        p_motion: float,
    ) -> float:
        """Voting consensus fusion across streams above 0.5 confidence threshold."""
        votes = sum([1 for p in [p_graph, p_action, p_motion] if p >= 0.50])
        if votes == 3:
            return float(np.mean([p_graph, p_action, p_motion]) + 0.10)
        elif votes == 2:
            return float(np.mean([p_graph, p_action, p_motion]))
        else:
            return float(np.mean([p_graph, p_action, p_motion]) - 0.10)

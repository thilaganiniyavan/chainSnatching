"""Behaviour Reasoning Engine — composing primitives into higher-level events.

Consumes completed Behaviour Timelines and classifies each interaction into
zero or more :class:`BehaviourEvent` instances by evaluating a configurable
rule graph.  Every decision is explainable via the evidence chain attached
to each event.

The engine is independent from the future Snatch Signature Engine; it produces
domain-general Behaviour Events that the signature engine will later consume.
"""

from __future__ import annotations

from typing import Any

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_event import BehaviourEvent
from src.behavior.behaviour_timeline import TimelineEvent
from src.behavior.reasoning_rules import RuleNode, get_default_rules
from src.behavior.explanation_generator import ExplanationGenerator


class ReasoningEngine:
    """Evaluates completed interaction timelines against a rule graph
    to produce higher-level Behaviour Events.

    Args:
        rules: Custom list of :class:`RuleNode` instances.  If ``None``,
            the default 7-event rule set is used.
        fps: Video frame rate for duration conversion.
    """

    def __init__(
        self,
        rules: list[RuleNode] | None = None,
        fps: float = 30.0,
    ) -> None:
        self.rules: list[RuleNode] = rules if rules is not None else get_default_rules()
        self.fps: float = fps if fps > 0 else 30.0
        self._explainer = ExplanationGenerator()
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_interaction(
        self,
        interaction: Interaction,
        timeline: list[TimelineEvent],
        *,
        tentative: bool = False,
    ) -> list[BehaviourEvent]:
        """Classify a single interaction's timeline into Behaviour Events.

        Args:
            interaction: The interaction object (completed or active).
            timeline: Chronological timeline events for this interaction.
            tentative: If ``True``, marks output events as tentative
                (used for real-time estimation on active interactions).

        Returns:
            A list of :class:`BehaviourEvent` instances (may be empty).
        """
        # Step 1 — Extract primitive sequence (ignore lifecycle events)
        primitive_sequence = [
            evt.event_type
            for evt in timeline
            if not evt.event_type.startswith("INTERACTION_")
        ]

        if not primitive_sequence:
            return []

        # Step 2 — Compute aggregate evidence
        motion_evidence = self._compute_motion_evidence(interaction)
        spatial_evidence = self._compute_spatial_evidence(interaction)

        # Step 3 — Evaluate every rule
        candidates: list[tuple[float, int, BehaviourEvent]] = []

        for rule in self.rules:
            match_result = self._evaluate_rule(
                rule, interaction, primitive_sequence,
                motion_evidence, spatial_evidence, tentative,
            )
            if match_result is not None:
                # Tuple of (confidence, priority, event) for sorting
                candidates.append(
                    (match_result.confidence, rule.priority, match_result)
                )

        if not candidates:
            return []

        # Step 4 — Priority-based conflict resolution
        # Sort by priority descending, then confidence descending
        candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)

        # Return the highest-priority event.  If there are multiple events
        # at the same priority level, return all of them.
        top_priority = candidates[0][1]
        events = [
            c[2] for c in candidates
            if c[1] == top_priority
        ]

        return events

    def analyse_all(
        self,
        interactions: list[Interaction],
        timelines: dict[str, list[TimelineEvent]],
        *,
        tentative: bool = False,
    ) -> list[BehaviourEvent]:
        """Classify multiple interactions at once.

        Args:
            interactions: List of interaction objects.
            timelines: Mapping from interaction_id to timeline events.
            tentative: Passed through to :meth:`analyse_interaction`.

        Returns:
            Flat list of all produced :class:`BehaviourEvent` instances.
        """
        all_events: list[BehaviourEvent] = []
        for interaction in interactions:
            tl = timelines.get(interaction.interaction_id, [])
            events = self.analyse_interaction(
                interaction, tl, tentative=tentative
            )
            all_events.extend(events)
        return all_events

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: RuleNode,
        interaction: Interaction,
        primitive_sequence: list[str],
        motion_evidence: dict[str, Any],
        spatial_evidence: dict[str, Any],
        tentative: bool,
    ) -> BehaviourEvent | None:
        """Evaluate a single rule against the interaction evidence.

        Returns a :class:`BehaviourEvent` if the rule matches, else ``None``.
        """
        # --- Primitive check ---
        matched_primitives = [
            p for p in rule.required_primitives
            if p in primitive_sequence
        ]
        if len(matched_primitives) < rule.min_primitive_count:
            return None

        # --- Sequence ordering check ---
        if rule.required_sequence is not None:
            if not self._check_sequence_order(
                primitive_sequence, rule.required_sequence
            ):
                return None

        # --- Temporal constraints ---
        duration = interaction.duration
        if duration < rule.min_duration_frames:
            return None
        if rule.max_duration_frames is not None and duration > rule.max_duration_frames:
            return None

        # --- Spatial constraints ---
        min_dist = spatial_evidence.get("min_distance", float("inf"))
        if rule.max_distance is not None and min_dist > rule.max_distance:
            return None

        if rule.min_distance is not None and min_dist < rule.min_distance:
            return None

        # --- Kinematic constraints ---
        peak_velocity = motion_evidence.get("peak_relative_velocity", 0.0)
        peak_acceleration = motion_evidence.get("peak_relative_acceleration", 0.0)

        if rule.min_velocity is not None and peak_velocity < rule.min_velocity:
            return None
        if rule.max_velocity is not None and peak_velocity > rule.max_velocity:
            return None
        if rule.min_acceleration is not None and peak_acceleration < rule.min_acceleration:
            return None
        if rule.max_acceleration is not None and peak_acceleration > rule.max_acceleration:
            return None

        # --- All checks passed — compute confidence ---
        confidence = self._compute_confidence(
            rule, matched_primitives, primitive_sequence,
            interaction, motion_evidence, spatial_evidence,
        )

        # Lower confidence for tentative (active) interactions
        if tentative:
            confidence *= 0.7

        # --- Build the supporting sequence (only matched primitives in order) ---
        supporting = self._extract_ordered_matches(
            primitive_sequence, rule.required_primitives
        )

        # --- Generate explanation ---
        explanation = self._explainer.generate(
            rule.event_type, interaction, motion_evidence,
            spatial_evidence, supporting, self.fps,
        )

        # --- Build event ---
        self._next_id += 1
        duration_seconds = round(duration / self.fps, 3)

        event = BehaviourEvent(
            event_id=f"EVT-{self._next_id:04d}",
            event_type=rule.event_type,
            confidence=round(confidence, 4),
            start_frame=interaction.start_frame,
            end_frame=interaction.end_frame or interaction.current_frame,
            duration_frames=duration,
            duration_seconds=duration_seconds,
            participants={
                "person_track_id": interaction.person_track_id,
                "vehicle_track_id": interaction.vehicle_track_id,
            },
            interaction_id=interaction.interaction_id,
            supporting_sequence=supporting,
            motion_evidence=motion_evidence,
            spatial_evidence=spatial_evidence,
            explanation=explanation,
            is_tentative=tentative,
            metadata={
                "rule_event_type": rule.event_type,
                "rule_priority": rule.priority,
                "matched_primitives": matched_primitives,
                "rule_description": rule.description,
            },
        )

        return event

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        rule: RuleNode,
        matched_primitives: list[str],
        full_sequence: list[str],
        interaction: Interaction,
        motion_evidence: dict[str, Any],
        spatial_evidence: dict[str, Any],
    ) -> float:
        """Compute classification confidence from evidence strength.

        Starts at ``rule.base_confidence`` and adjusts based on:
        - Primitive match ratio
        - Duration fit
        - Spatial evidence strength
        - Kinematic evidence strength
        """
        conf = rule.base_confidence

        # 1. Primitive match ratio bonus (0 to 0.2)
        total_required = len(rule.required_primitives)
        if total_required > 0:
            match_ratio = len(matched_primitives) / total_required
            conf += 0.2 * match_ratio

        # 2. Duration fit bonus (0 to 0.1)
        duration = interaction.duration
        if rule.max_duration_frames is not None:
            ideal_mid = (rule.min_duration_frames + rule.max_duration_frames) / 2.0
            if ideal_mid > 0:
                fit = 1.0 - min(abs(duration - ideal_mid) / ideal_mid, 1.0)
                conf += 0.1 * fit
        else:
            # Open-ended duration — longer is stronger
            fit = min(duration / 60.0, 1.0)
            conf += 0.1 * fit

        # 3. Spatial evidence bonus (0 to 0.1)
        min_dist = spatial_evidence.get("min_distance", float("inf"))
        if rule.max_distance is not None and rule.max_distance > 0:
            closeness = max(0.0, 1.0 - min_dist / rule.max_distance)
            conf += 0.1 * closeness

        # 4. Kinematic evidence bonus (0 to 0.1)
        peak_accel = motion_evidence.get("peak_relative_acceleration", 0.0)
        if rule.min_acceleration is not None and rule.min_acceleration > 0:
            accel_strength = min(peak_accel / (rule.min_acceleration * 3), 1.0)
            conf += 0.1 * accel_strength

        return min(conf, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sequence_order(
        full_sequence: list[str],
        required_order: list[str],
    ) -> bool:
        """Check that *required_order* elements appear in order within
        *full_sequence* (gaps between them are allowed)."""
        idx = 0
        for primitive in full_sequence:
            if idx < len(required_order) and primitive == required_order[idx]:
                idx += 1
        return idx == len(required_order)

    @staticmethod
    def _extract_ordered_matches(
        full_sequence: list[str],
        required_primitives: list[str],
    ) -> list[str]:
        """Extract the first occurrence of each required primitive
        in timeline order (preserving the order they appeared)."""
        required_set = set(required_primitives)
        seen: set[str] = set()
        result: list[str] = []
        for p in full_sequence:
            if p in required_set and p not in seen:
                result.append(p)
                seen.add(p)
        return result

    @staticmethod
    def _compute_motion_evidence(interaction: Interaction) -> dict[str, Any]:
        """Aggregate kinematic measurements from the interaction."""
        person_speeds: list[float] = []
        vehicle_speeds: list[float] = []
        velocities: list[float] = []
        accelerations: list[float] = []

        for snap in interaction.motion_history:
            ps = snap.get("person_speed") or 0.0
            vs = snap.get("vehicle_speed") or 0.0
            person_speeds.append(ps)
            vehicle_speeds.append(vs)

        # Compute velocity and acceleration from relationship history
        for i, rh in enumerate(interaction.relationship_history):
            dist = rh.get("distance", 0.0)
            if i > 0:
                prev_dist = interaction.relationship_history[i - 1].get("distance", 0.0)
                vel = dist - prev_dist
                velocities.append(vel)
                if len(velocities) >= 2:
                    accel = velocities[-1] - velocities[-2]
                    accelerations.append(accel)

        return {
            "person_avg_speed": round(
                sum(person_speeds) / max(len(person_speeds), 1), 4
            ),
            "person_max_speed": round(
                max(person_speeds, default=0.0), 4
            ),
            "vehicle_avg_speed": round(
                sum(vehicle_speeds) / max(len(vehicle_speeds), 1), 4
            ),
            "vehicle_max_speed": round(
                max(vehicle_speeds, default=0.0), 4
            ),
            "peak_relative_velocity": round(
                max((abs(v) for v in velocities), default=0.0), 4
            ),
            "avg_relative_velocity": round(
                sum(velocities) / max(len(velocities), 1), 4
            ) if velocities else 0.0,
            "peak_relative_acceleration": round(
                max((abs(a) for a in accelerations), default=0.0), 4
            ),
            "final_relative_velocity": round(
                interaction.relative_velocity, 4
            ),
            "final_relative_acceleration": round(
                interaction.relative_acceleration, 4
            ),
        }

    @staticmethod
    def _compute_spatial_evidence(interaction: Interaction) -> dict[str, Any]:
        """Aggregate spatial measurements from the interaction."""
        distances = [
            rh.get("distance", 0.0)
            for rh in interaction.relationship_history
        ]

        return {
            "min_distance": round(interaction.min_distance, 4),
            "max_distance": round(interaction.max_distance, 4),
            "avg_distance": round(interaction.avg_distance, 4),
            "final_distance": round(interaction.current_distance, 4),
            "distance_samples": len(distances),
        }

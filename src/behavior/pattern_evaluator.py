"""Pattern Evaluator — converts primitive sequences and kinematics into Behaviour Patterns.

Maps primitive behaviours, interaction lifecycle states, and quantitative
motion/spatial measurements into 11 reusable Behaviour Patterns.

Every pattern detector is an independent method returning an Optional[PatternNode],
allowing isolated unit testing and clean modular design.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models.behaviour_graph import PatternNode
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.core.models.interaction import Interaction, InteractionState
from src.behavior.behaviour_timeline import TimelineEvent
from src.behavior.pattern_rules import PatternConfig


class PatternEvaluator:
    """Evaluates combinations of primitive behaviours, interaction state,
    and quantitative kinematics to generate PatternNode instances.

    Args:
        config: Custom PatternConfig instance. If None, default thresholds are used.
        fps: Video FPS for converting frame durations to seconds.
    """

    def __init__(
        self,
        config: PatternConfig | None = None,
        fps: float = 30.0,
    ) -> None:
        self.config = config if config is not None else PatternConfig()
        self.fps = fps if fps > 0 else 30.0

    def evaluate(
        self,
        interaction: Interaction,
        primitives: list[BehaviourPrimitive],
        timeline: list[TimelineEvent],
        frame_number: int,
    ) -> list[PatternNode]:
        """Evaluate interaction and primitive history to emit active PatternNodes.

        Args:
            interaction: Current Interaction domain model.
            primitives: BehaviourPrimitive objects detected in current frame.
            timeline: Accumulated TimelineEvent list for this interaction.
            frame_number: Current video frame number.

        Returns:
            List of detected :class:`PatternNode` instances for the current window.
        """
        if interaction.state == InteractionState.ARCHIVED:
            return []

        prim_types = set(bp.primitive_type for bp in primitives)
        timeline_prims = [
            e.event_type for e in timeline
            if not e.event_type.startswith("INTERACTION_")
        ]

        motion_stats = self._extract_motion_stats(interaction)
        spatial_stats = self._extract_spatial_stats(interaction)

        detectors = [
            self._eval_approach_pattern,
            self._eval_follow_pattern,
            self._eval_co_travel_pattern,
            self._eval_proximity_pattern,
            self._eval_interaction_pattern,
            self._eval_stop_pattern,
            self._eval_lingering_pattern,
            self._eval_separation_pattern,
            self._eval_escape_pattern,
            self._eval_divergence_pattern,
            self._eval_waiting_pattern,
        ]

        patterns: list[PatternNode] = []
        for detector in detectors:
            node = detector(
                interaction, prim_types, timeline_prims,
                motion_stats, spatial_stats, frame_number,
            )
            if node is not None:
                patterns.append(node)

        return patterns

    # ------------------------------------------------------------------
    # Individual Pattern Detectors
    # ------------------------------------------------------------------

    def _eval_approach_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """APPROACH_PATTERN: Closing velocity + decreasing distance + APPROACHING primitive."""
        if (
            "APPROACHING" in prim_types or "APPROACHING" in timeline_prims
        ) and interaction.relative_velocity < self.config.approach_velocity_threshold:

            conf = min(
                self.config.base_confidence
                + abs(interaction.relative_velocity) / 20.0,
                1.0,
            )
            return self._build_node(
                pattern_type="APPROACH_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=8,
                primitives=["APPROACHING"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_follow_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """FOLLOW_PATTERN: FOLLOWING primitive or aligned trajectory similarity + min duration."""
        if (
            "FOLLOWING" in prim_types
            or "FOLLOWING" in timeline_prims
            or (
                interaction.trajectory_similarity > self.config.trajectory_similarity_threshold
                and interaction.heading_difference < self.config.heading_parallel_threshold * 2
            )
        ) and interaction.duration >= self.config.min_follow_frames:

            conf = min(
                self.config.base_confidence
                + interaction.trajectory_similarity * 0.3,
                1.0,
            )
            return self._build_node(
                pattern_type="FOLLOW_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=7,
                primitives=["FOLLOWING"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_co_travel_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """CO_TRAVEL_PATTERN: PARALLEL_MOTION primitive + parallel headings + non-zero speed."""
        if (
            "PARALLEL_MOTION" in prim_types or "PARALLEL_MOTION" in timeline_prims
        ) and interaction.heading_difference < self.config.heading_parallel_threshold:

            conf = min(
                self.config.base_confidence
                + (1.0 - interaction.heading_difference / 180.0) * 0.3,
                1.0,
            )
            return self._build_node(
                pattern_type="CO_TRAVEL_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=7,
                primitives=["PARALLEL_MOTION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_proximity_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """PROXIMITY_PATTERN: CLOSE_INTERACTION primitive or current distance below threshold."""
        if (
            "CLOSE_INTERACTION" in prim_types
            or interaction.current_distance < self.config.proximity_distance_threshold
        ):
            conf = min(
                self.config.base_confidence
                + (1.0 - interaction.current_distance / self.config.proximity_distance_threshold) * 0.4,
                1.0,
            )
            return self._build_node(
                pattern_type="PROXIMITY_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=3,
                primitives=["CLOSE_INTERACTION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_interaction_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """INTERACTION_PATTERN: Sustained close proximity under threshold + multi-frame history."""
        if (
            interaction.min_distance < self.config.close_interaction_distance_threshold
            and interaction.duration >= self.config.min_pattern_frames
        ):
            conf = min(
                self.config.base_confidence
                + (1.0 - interaction.min_distance / self.config.close_interaction_distance_threshold) * 0.35,
                1.0,
            )
            return self._build_node(
                pattern_type="INTERACTION_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=6,
                primitives=["CLOSE_INTERACTION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_stop_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """STOP_PATTERN: STATIONARY_INTERACTION primitive or near-zero speeds."""
        p_speed = motion.get("person_avg_speed", 0.0)
        v_speed = motion.get("vehicle_avg_speed", 0.0)

        if (
            "STATIONARY_INTERACTION" in prim_types
            or (p_speed < self.config.stationary_speed_threshold and v_speed < self.config.stationary_speed_threshold)
        ) and interaction.current_distance < self.config.proximity_distance_threshold:

            conf = min(self.config.base_confidence + 0.3, 1.0)
            return self._build_node(
                pattern_type="STOP_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=4,
                primitives=["STATIONARY_INTERACTION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_lingering_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """LINGERING_PATTERN: Interaction state is LINGERING or prolonged low-speed proximity."""
        if (
            interaction.state == InteractionState.LINGERING
            or interaction.duration >= self.config.min_lingering_frames
        ) and interaction.avg_distance < self.config.proximity_distance_threshold:

            conf = min(self.config.base_confidence + 0.25, 1.0)
            return self._build_node(
                pattern_type="LINGERING_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=1,
                primitives=["INTERACTION_DURATION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_separation_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """SEPARATION_PATTERN: MOVING_AWAY primitive + positive opening velocity."""
        if (
            "MOVING_AWAY" in prim_types or "MOVING_AWAY" in timeline_prims
        ) and interaction.relative_velocity > self.config.separation_velocity_threshold:

            conf = min(
                self.config.base_confidence
                + interaction.relative_velocity / 15.0,
                1.0,
            )
            return self._build_node(
                pattern_type="SEPARATION_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=8,
                primitives=["MOVING_AWAY"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_escape_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """ESCAPE_PATTERN: RAPID_ACCELERATION + RAPID_SEPARATION primitives or high accel/vel."""
        peak_acc = motion.get("peak_relative_acceleration", 0.0)
        peak_vel = motion.get("peak_relative_velocity", 0.0)

        if (
            ("RAPID_ACCELERATION" in prim_types and "RAPID_SEPARATION" in prim_types)
            or ("RAPID_ACCELERATION" in timeline_prims and "RAPID_SEPARATION" in timeline_prims)
            or (peak_acc >= self.config.escape_acceleration_threshold and peak_vel >= self.config.escape_velocity_threshold)
        ):
            conf = min(
                self.config.base_confidence + 0.35,
                1.0,
            )
            return self._build_node(
                pattern_type="ESCAPE_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=10,
                primitives=["RAPID_ACCELERATION", "RAPID_SEPARATION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_divergence_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """DIVERGENCE_PATTERN: SUDDEN_DIRECTION_CHANGE primitive + diverging heading angle."""
        if (
            "SUDDEN_DIRECTION_CHANGE" in prim_types or "SUDDEN_DIRECTION_CHANGE" in timeline_prims
        ) and interaction.heading_difference > self.config.heading_divergence_threshold:

            conf = min(
                self.config.base_confidence
                + interaction.heading_difference / 180.0 * 0.3,
                1.0,
            )
            return self._build_node(
                pattern_type="DIVERGENCE_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=9,
                primitives=["SUDDEN_DIRECTION_CHANGE"],
                motion=motion,
                spatial=spatial,
            )
        return None

    def _eval_waiting_pattern(
        self,
        interaction: Interaction,
        prim_types: set[str],
        timeline_prims: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
        frame_number: int,
    ) -> Optional[PatternNode]:
        """WAITING_PATTERN: Stationary vehicle near pedestrian over minimum waiting duration."""
        v_speed = motion.get("vehicle_avg_speed", 0.0)

        if (
            "STATIONARY_INTERACTION" in prim_types or "STATIONARY_INTERACTION" in timeline_prims
        ) and v_speed < self.config.stationary_speed_threshold and interaction.duration >= self.config.min_waiting_frames:

            conf = min(self.config.base_confidence + 0.25, 1.0)
            return self._build_node(
                pattern_type="WAITING_PATTERN",
                interaction=interaction,
                frame_number=frame_number,
                confidence=conf,
                priority=5,
                primitives=["STATIONARY_INTERACTION"],
                motion=motion,
                spatial=spatial,
            )
        return None

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _build_node(
        self,
        pattern_type: str,
        interaction: Interaction,
        frame_number: int,
        confidence: float,
        priority: int,
        primitives: list[str],
        motion: dict[str, Any],
        spatial: dict[str, Any],
    ) -> PatternNode:
        """Helper to construct a standard PatternNode."""
        dur_frames = interaction.duration
        dur_sec = round(dur_frames / self.fps, 3)

        return PatternNode(
            pattern_type=pattern_type,
            start_frame=interaction.start_frame,
            end_frame=frame_number,
            duration_frames=dur_frames,
            duration_seconds=dur_sec,
            confidence=round(confidence, 4),
            priority=priority,
            supporting_primitives=primitives,
            supporting_motion=motion,
            supporting_spatial=spatial,
            extensible_evidence={
                "interaction_state": interaction.state.value,
                "person_track_id": interaction.person_track_id,
                "vehicle_track_id": interaction.vehicle_track_id,
                "pose_skeleton_data": None,     # Extensibility hook for future ST-GCN
                "fusion_embeddings": None,     # Extensibility hook for future Fusion
            },
        )

    @staticmethod
    def _extract_motion_stats(interaction: Interaction) -> dict[str, Any]:
        """Extract motion statistics from interaction history."""
        person_speeds = [
            m.get("person_speed", 0.0) or 0.0
            for m in interaction.motion_history
        ]
        vehicle_speeds = [
            m.get("vehicle_speed", 0.0) or 0.0
            for m in interaction.motion_history
        ]
        return {
            "person_avg_speed": round(sum(person_speeds) / max(len(person_speeds), 1), 4),
            "vehicle_avg_speed": round(sum(vehicle_speeds) / max(len(vehicle_speeds), 1), 4),
            "relative_velocity": round(interaction.relative_velocity, 4),
            "relative_acceleration": round(interaction.relative_acceleration, 4),
            "heading_difference": round(interaction.heading_difference, 4),
            "trajectory_similarity": round(interaction.trajectory_similarity, 4),
            "peak_relative_velocity": round(abs(interaction.relative_velocity), 4),
            "peak_relative_acceleration": round(abs(interaction.relative_acceleration), 4),
        }

    @staticmethod
    def _extract_spatial_stats(interaction: Interaction) -> dict[str, Any]:
        """Extract spatial statistics from interaction history."""
        return {
            "min_distance": round(interaction.min_distance, 4),
            "max_distance": round(interaction.max_distance, 4),
            "avg_distance": round(interaction.avg_distance, 4),
            "current_distance": round(interaction.current_distance, 4),
        }

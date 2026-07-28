"""Behaviour Intelligence Engine — behavioural primitive extraction.

Consumes :class:`Interaction` objects (never raw tracks) and classifies each
interaction into zero or more :class:`BehaviourPrimitive` instances per frame.

Every detector is an independent private method returning
``Optional[BehaviourPrimitive]``, making each primitive independently testable.

All thresholds are constructor-injected so the engine can be reconfigured
without code changes — a requirement for ablation studies.
"""

from __future__ import annotations

from typing import Optional

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive


class BehaviourEngine:
    """Stateless per-frame classifier that extracts behavioural primitives
    from Interaction objects.

    Configuration:
        close_distance_threshold: Distance (px) below which objects are "close".
        approach_velocity_threshold: Closing rate (px/frame) to flag approach.
        separation_velocity_threshold: Opening rate to flag rapid separation.
        acceleration_threshold: Absolute accel to flag rapid accel/decel.
        heading_change_threshold: Degrees per frame for sudden direction change.
        following_similarity_threshold: Cosine similarity for "following".
        following_min_frames: Minimum duration to call "following".
        stationary_speed_threshold: Speed (px/frame) below which an object is still.
        parallel_heading_threshold: Heading diff (degrees) for parallel motion.
        parallel_distance_stability: Max distance variance for stable distance.
        duration_bucket_size: Frames per "INTERACTION_DURATION" bucket emission.
    """

    def __init__(
        self,
        close_distance_threshold: float = 80.0,
        approach_velocity_threshold: float = -2.0,
        separation_velocity_threshold: float = 5.0,
        acceleration_threshold: float = 3.0,
        heading_change_threshold: float = 45.0,
        following_similarity_threshold: float = 0.7,
        following_min_frames: int = 5,
        stationary_speed_threshold: float = 2.0,
        parallel_heading_threshold: float = 15.0,
        parallel_distance_stability: float = 20.0,
        duration_bucket_size: int = 30,
    ) -> None:
        self.close_distance_threshold = close_distance_threshold
        self.approach_velocity_threshold = approach_velocity_threshold
        self.separation_velocity_threshold = separation_velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.heading_change_threshold = heading_change_threshold
        self.following_similarity_threshold = following_similarity_threshold
        self.following_min_frames = following_min_frames
        self.stationary_speed_threshold = stationary_speed_threshold
        self.parallel_heading_threshold = parallel_heading_threshold
        self.parallel_distance_stability = parallel_distance_stability
        self.duration_bucket_size = duration_bucket_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        interactions: list[Interaction],
        frame_number: int,
    ) -> list[BehaviourPrimitive]:
        """Extract behavioural primitives from all active interactions.

        Args:
            interactions: Current interaction objects (any lifecycle state).
            frame_number: The current video frame number.

        Returns:
            A list of detected :class:`BehaviourPrimitive` instances.
        """
        primitives: list[BehaviourPrimitive] = []

        for interaction in interactions:
            # Only analyse interactions that are still evolving
            if interaction.state in (
                InteractionState.ENDED,
                InteractionState.ARCHIVED,
            ):
                continue

            detectors = [
                self._detect_approaching,
                self._detect_moving_away,
                self._detect_following,
                self._detect_parallel_motion,
                self._detect_close_interaction,
                self._detect_stationary_interaction,
                self._detect_rapid_acceleration,
                self._detect_rapid_deceleration,
                self._detect_sudden_direction_change,
                self._detect_rapid_separation,
                self._detect_interaction_duration,
            ]

            for detector in detectors:
                result = detector(interaction, frame_number)
                if result is not None:
                    primitives.append(result)

        return primitives

    # ------------------------------------------------------------------
    # Individual primitive detectors
    # ------------------------------------------------------------------

    def _detect_approaching(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the person and vehicle are closing distance."""
        if (
            interaction.relative_velocity < self.approach_velocity_threshold
            and interaction.current_distance > self.close_distance_threshold
        ):
            confidence = min(
                abs(interaction.relative_velocity)
                / abs(self.approach_velocity_threshold * 5),
                1.0,
            )
            return BehaviourPrimitive(
                primitive_type="APPROACHING",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "relative_velocity": interaction.relative_velocity,
                    "current_distance": interaction.current_distance,
                },
            )
        return None

    def _detect_moving_away(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the person and vehicle are separating."""
        if (
            interaction.relative_velocity > 0
            and interaction.current_distance > self.close_distance_threshold
        ):
            confidence = min(
                interaction.relative_velocity / self.separation_velocity_threshold,
                1.0,
            )
            return BehaviourPrimitive(
                primitive_type="MOVING_AWAY",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(max(confidence, 0.0), 4),
                measurements={
                    "relative_velocity": interaction.relative_velocity,
                    "current_distance": interaction.current_distance,
                },
            )
        return None

    def _detect_following(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the vehicle is following the person (or vice versa)."""
        if (
            interaction.heading_difference < self.parallel_heading_threshold * 2
            and interaction.trajectory_similarity
            > self.following_similarity_threshold
            and interaction.duration >= self.following_min_frames
        ):
            confidence = (
                interaction.trajectory_similarity * 0.6
                + min(interaction.duration / 30.0, 1.0) * 0.4
            )
            return BehaviourPrimitive(
                primitive_type="FOLLOWING",
                interaction_id=interaction.interaction_id,
                start_frame=interaction.start_frame,
                end_frame=frame_number,
                confidence=round(min(confidence, 1.0), 4),
                measurements={
                    "heading_difference": interaction.heading_difference,
                    "trajectory_similarity": interaction.trajectory_similarity,
                    "duration": interaction.duration,
                },
            )
        return None

    def _detect_parallel_motion(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect near-parallel movement with stable inter-object distance."""
        if interaction.duration < 3:
            return None

        distance_range = interaction.max_distance - interaction.min_distance

        if (
            interaction.heading_difference < self.parallel_heading_threshold
            and distance_range < self.parallel_distance_stability
        ):
            confidence = (
                (1.0 - interaction.heading_difference / 180.0) * 0.5
                + (1.0 - min(distance_range / self.parallel_distance_stability, 1.0))
                * 0.5
            )
            return BehaviourPrimitive(
                primitive_type="PARALLEL_MOTION",
                interaction_id=interaction.interaction_id,
                start_frame=interaction.start_frame,
                end_frame=frame_number,
                confidence=round(min(confidence, 1.0), 4),
                measurements={
                    "heading_difference": interaction.heading_difference,
                    "distance_range": distance_range,
                    "avg_distance": interaction.avg_distance,
                },
            )
        return None

    def _detect_close_interaction(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the two objects are within the close-distance threshold."""
        if interaction.current_distance < self.close_distance_threshold:
            confidence = 1.0 - (
                interaction.current_distance / self.close_distance_threshold
            )
            return BehaviourPrimitive(
                primitive_type="CLOSE_INTERACTION",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(max(confidence, 0.0), 4),
                measurements={
                    "current_distance": interaction.current_distance,
                    "threshold": self.close_distance_threshold,
                },
            )
        return None

    def _detect_stationary_interaction(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when both objects are near-stationary and close."""
        if not interaction.motion_history:
            return None

        latest = interaction.motion_history[-1]
        p_speed = latest.get("person_speed") or 0.0
        v_speed = latest.get("vehicle_speed") or 0.0

        if (
            p_speed < self.stationary_speed_threshold
            and v_speed < self.stationary_speed_threshold
            and interaction.current_distance < self.close_distance_threshold
        ):
            speed_score = 1.0 - max(p_speed, v_speed) / self.stationary_speed_threshold
            dist_score = 1.0 - interaction.current_distance / self.close_distance_threshold
            confidence = speed_score * 0.5 + dist_score * 0.5
            return BehaviourPrimitive(
                primitive_type="STATIONARY_INTERACTION",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(max(confidence, 0.0), 4),
                measurements={
                    "person_speed": p_speed,
                    "vehicle_speed": v_speed,
                    "current_distance": interaction.current_distance,
                },
            )
        return None

    def _detect_rapid_acceleration(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the relative acceleration exceeds the threshold."""
        if interaction.relative_acceleration > self.acceleration_threshold:
            confidence = min(
                interaction.relative_acceleration / (self.acceleration_threshold * 3),
                1.0,
            )
            return BehaviourPrimitive(
                primitive_type="RAPID_ACCELERATION",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "relative_acceleration": interaction.relative_acceleration,
                    "relative_velocity": interaction.relative_velocity,
                },
            )
        return None

    def _detect_rapid_deceleration(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect when the relative deceleration exceeds the threshold."""
        if interaction.relative_acceleration < -self.acceleration_threshold:
            confidence = min(
                abs(interaction.relative_acceleration)
                / (self.acceleration_threshold * 3),
                1.0,
            )
            return BehaviourPrimitive(
                primitive_type="RAPID_DECELERATION",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "relative_acceleration": interaction.relative_acceleration,
                    "relative_velocity": interaction.relative_velocity,
                },
            )
        return None

    def _detect_sudden_direction_change(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect a sudden heading change between consecutive frames."""
        if len(interaction.motion_history) < 2:
            return None

        prev = interaction.motion_history[-2]
        curr = interaction.motion_history[-1]

        prev_heading = self._safe_heading_diff(
            prev.get("person_direction"), prev.get("vehicle_direction")
        )
        curr_heading = self._safe_heading_diff(
            curr.get("person_direction"), curr.get("vehicle_direction")
        )

        if prev_heading is None or curr_heading is None:
            return None

        delta = abs(curr_heading - prev_heading)
        if delta > 180:
            delta = 360 - delta

        if delta > self.heading_change_threshold:
            confidence = min(delta / (self.heading_change_threshold * 3), 1.0)
            return BehaviourPrimitive(
                primitive_type="SUDDEN_DIRECTION_CHANGE",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "heading_delta": delta,
                    "prev_heading_diff": prev_heading,
                    "curr_heading_diff": curr_heading,
                },
            )
        return None

    def _detect_rapid_separation(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Detect rapid increase in distance between the two objects."""
        if (
            interaction.relative_velocity > self.separation_velocity_threshold
            and interaction.current_distance > interaction.avg_distance
        ):
            confidence = min(
                interaction.relative_velocity
                / (self.separation_velocity_threshold * 3),
                1.0,
            )
            return BehaviourPrimitive(
                primitive_type="RAPID_SEPARATION",
                interaction_id=interaction.interaction_id,
                start_frame=frame_number,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "relative_velocity": interaction.relative_velocity,
                    "current_distance": interaction.current_distance,
                    "avg_distance": interaction.avg_distance,
                },
            )
        return None

    def _detect_interaction_duration(
        self, interaction: Interaction, frame_number: int
    ) -> Optional[BehaviourPrimitive]:
        """Emit periodic duration milestone events."""
        if (
            self.duration_bucket_size > 0
            and interaction.duration > 0
            and interaction.duration % self.duration_bucket_size == 0
        ):
            bucket = interaction.duration // self.duration_bucket_size
            confidence = min(bucket / 10.0, 1.0)
            return BehaviourPrimitive(
                primitive_type="INTERACTION_DURATION",
                interaction_id=interaction.interaction_id,
                start_frame=interaction.start_frame,
                end_frame=frame_number,
                confidence=round(confidence, 4),
                measurements={
                    "duration_frames": interaction.duration,
                    "bucket": bucket,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_heading_diff(
        dir_a: float | None, dir_b: float | None
    ) -> float | None:
        """Compute angular difference; return None if either direction is missing."""
        if dir_a is None or dir_b is None:
            return None
        diff = abs(dir_a - dir_b) % 360
        return diff if diff <= 180 else 360 - diff

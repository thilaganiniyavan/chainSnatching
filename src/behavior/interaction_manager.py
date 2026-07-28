"""Interaction Manager for persistent person-vehicle encounter tracking.

Converts per-frame relationship snapshots into stateful Interaction objects
with full lifecycle management.  Each interaction maintains spatial, kinematic,
and temporal measurements across its lifetime.

The manager is instantiated once and called every frame via ``update()``.
"""

from __future__ import annotations

import math
from typing import Optional

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.relationship import Relationship
from src.core.models.track import Track


class InteractionManager:
    """Creates, updates, and manages the lifecycle of Interaction objects.

    Configuration:
        distance_threshold: Maximum distance (px) to initiate a new interaction.
        linger_frames: Frames without proximity before ACTIVE -> LINGERING.
        end_frames: Frames in LINGERING before transitioning to ENDED.
        archive_after_frames: Frames after ENDED before ARCHIVED (0 = immediate).
        confidence_base: Base weight for the confidence scorer.
    """

    def __init__(
        self,
        distance_threshold: float = 150.0,
        linger_frames: int = 10,
        end_frames: int = 30,
        archive_after_frames: int = 0,
        confidence_base: float = 0.5,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.linger_frames = linger_frames
        self.end_frames = end_frames
        self.archive_after_frames = archive_after_frames
        self.confidence_base = confidence_base

        # Internal registries
        self._interactions: dict[str, Interaction] = {}
        # Maps (person_track_id, vehicle_track_id) -> interaction_id
        self._pair_index: dict[tuple[int, int], str] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        person_track: Track,
        vehicle_track: Track,
        frame_number: int,
        distance: float,
    ) -> Interaction:
        """Create a new Interaction for a person-vehicle pair.

        Returns the newly created Interaction object.
        """
        self._next_id += 1
        interaction = Interaction(
            interaction_id=f"INT-{self._next_id:04d}",
            person_track_id=person_track.tracking_id,
            vehicle_track_id=vehicle_track.tracking_id,
            start_frame=frame_number,
            current_frame=frame_number,
            duration=1,
            min_distance=distance,
            max_distance=distance,
            avg_distance=distance,
            current_distance=distance,
            state=InteractionState.NEW,
            _distance_sum=distance,
            _distance_count=1,
        )

        # Snapshot initial motion state
        motion_snap = self._build_motion_snapshot(
            person_track, vehicle_track, frame_number
        )
        interaction.motion_history.append(motion_snap)

        # Snapshot initial relationship
        interaction.relationship_history.append(
            {
                "frame": frame_number,
                "distance": distance,
                "person_center": person_track.center,
                "vehicle_center": vehicle_track.center,
            }
        )

        self._interactions[interaction.interaction_id] = interaction
        pair_key = (person_track.tracking_id, vehicle_track.tracking_id)
        self._pair_index[pair_key] = interaction.interaction_id

        return interaction

    def update(
        self,
        relationships: list[Relationship],
        tracks: list[Track],
        frame_number: int,
    ) -> None:
        """Main per-frame entry point.

        1. Builds a lookup of current (person, vehicle) proximity pairs.
        2. Updates or creates interactions for each active pair.
        3. Ages interactions whose pair is absent this frame.
        """
        track_map: dict[int, Track] = {t.tracking_id: t for t in tracks}

        # Step 1 — determine which pairs are proximate this frame
        active_pairs: dict[tuple[int, int], Relationship] = {}
        for rel in relationships:
            pair_key = (rel.subject_id, rel.object_id)
            active_pairs[pair_key] = rel

        # Step 2 — update existing or create new interactions
        seen_interaction_ids: set[str] = set()

        for pair_key, rel in active_pairs.items():
            person_track = track_map.get(pair_key[0])
            vehicle_track = track_map.get(pair_key[1])
            if person_track is None or vehicle_track is None:
                continue

            if pair_key in self._pair_index:
                interaction_id = self._pair_index[pair_key]
                interaction = self._interactions.get(interaction_id)
                if interaction is not None and interaction.state not in (
                    InteractionState.ENDED,
                    InteractionState.ARCHIVED,
                ):
                    self._update_interaction(
                        interaction, person_track, vehicle_track,
                        rel.distance, frame_number,
                    )
                    seen_interaction_ids.add(interaction_id)
                else:
                    # Previous interaction ended; start a fresh one
                    new_int = self.create(
                        person_track, vehicle_track, frame_number, rel.distance
                    )
                    seen_interaction_ids.add(new_int.interaction_id)
            else:
                new_int = self.create(
                    person_track, vehicle_track, frame_number, rel.distance
                )
                seen_interaction_ids.add(new_int.interaction_id)

        # Step 3 — age unseen interactions
        for iid, interaction in list(self._interactions.items()):
            if iid in seen_interaction_ids:
                continue
            if interaction.state in (InteractionState.ENDED, InteractionState.ARCHIVED):
                self._maybe_archive(interaction, frame_number)
                continue

            interaction._frames_since_last_seen += 1

            if interaction.state in (InteractionState.NEW, InteractionState.ACTIVE):
                if interaction._frames_since_last_seen >= self.linger_frames:
                    interaction.state = InteractionState.LINGERING
            elif interaction.state == InteractionState.LINGERING:
                if interaction._frames_since_last_seen >= (
                    self.linger_frames + self.end_frames
                ):
                    self._end_interaction(interaction, frame_number)

    def terminate(self, interaction_id: str) -> None:
        """Manually terminate an interaction (force ENDED)."""
        interaction = self._interactions.get(interaction_id)
        if interaction is not None:
            interaction.state = InteractionState.ENDED
            interaction.end_frame = interaction.current_frame

    def get_active(self) -> list[Interaction]:
        """Return interactions in NEW or ACTIVE state."""
        return [
            i for i in self._interactions.values()
            if i.state in (InteractionState.NEW, InteractionState.ACTIVE)
        ]

    def get_by_state(self, state: InteractionState) -> list[Interaction]:
        """Return interactions matching a specific lifecycle state."""
        return [i for i in self._interactions.values() if i.state == state]

    def get_completed(self) -> list[Interaction]:
        """Return interactions in ENDED or ARCHIVED state."""
        return [
            i for i in self._interactions.values()
            if i.state in (InteractionState.ENDED, InteractionState.ARCHIVED)
        ]

    def get_all(self) -> list[Interaction]:
        """Return all tracked interactions regardless of state."""
        return list(self._interactions.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_interaction(
        self,
        interaction: Interaction,
        person_track: Track,
        vehicle_track: Track,
        distance: float,
        frame_number: int,
    ) -> None:
        """Update an existing interaction with this frame's measurements."""

        prev_distance = interaction.current_distance
        prev_velocity = interaction.relative_velocity

        # Temporal
        interaction.current_frame = frame_number
        interaction.duration = frame_number - interaction.start_frame + 1
        interaction._frames_since_last_seen = 0

        # Transition NEW -> ACTIVE after second observation
        if interaction.state == InteractionState.NEW:
            interaction.state = InteractionState.ACTIVE
        elif interaction.state == InteractionState.LINGERING:
            # Object re-appeared — reactivate
            interaction.state = InteractionState.ACTIVE

        # Spatial
        interaction.current_distance = distance
        interaction.min_distance = min(interaction.min_distance, distance)
        interaction.max_distance = max(interaction.max_distance, distance)
        interaction._distance_sum += distance
        interaction._distance_count += 1
        interaction.avg_distance = (
            interaction._distance_sum / interaction._distance_count
        )

        # Kinematics
        interaction.relative_velocity = distance - prev_distance
        if prev_velocity is not None:
            interaction.relative_acceleration = (
                interaction.relative_velocity - prev_velocity
            )
        interaction._previous_velocity = interaction.relative_velocity

        # Heading
        interaction.heading_difference = self._compute_heading_difference(
            person_track, vehicle_track
        )

        # Trajectory similarity
        interaction.trajectory_similarity = self._compute_trajectory_similarity(
            person_track, vehicle_track
        )

        # Confidence
        interaction.interaction_confidence = self._compute_confidence(interaction)

        # History snapshots
        interaction.relationship_history.append(
            {
                "frame": frame_number,
                "distance": distance,
                "person_center": person_track.center,
                "vehicle_center": vehicle_track.center,
            }
        )
        interaction.motion_history.append(
            self._build_motion_snapshot(person_track, vehicle_track, frame_number)
        )

    def _end_interaction(self, interaction: Interaction, frame_number: int) -> None:
        """Transition an interaction to ENDED."""
        interaction.state = InteractionState.ENDED
        interaction.end_frame = frame_number

    def _maybe_archive(self, interaction: Interaction, frame_number: int) -> None:
        """Transition ENDED interactions to ARCHIVED after the configured window."""
        if interaction.state != InteractionState.ENDED:
            return
        if interaction.end_frame is None:
            return
        if frame_number - interaction.end_frame >= self.archive_after_frames:
            interaction.state = InteractionState.ARCHIVED

    @staticmethod
    def _compute_heading_difference(
        person: Track, vehicle: Track
    ) -> float:
        """Angular difference between the two tracks' movement directions (degrees)."""
        p_dir = person.direction
        v_dir = vehicle.direction
        if p_dir is None or v_dir is None:
            return 0.0
        diff = abs(p_dir - v_dir) % 360
        return diff if diff <= 180 else 360 - diff

    @staticmethod
    def _compute_trajectory_similarity(
        person: Track, vehicle: Track
    ) -> float:
        """Cosine similarity of the two tracks' most recent displacement vectors.

        Returns 0.0 when insufficient history is available.
        """
        p_hist = person.history
        v_hist = vehicle.history
        if (
            p_hist is None
            or v_hist is None
            or len(p_hist.positions) < 2
            or len(v_hist.positions) < 2
        ):
            return 0.0

        px1, py1 = p_hist.positions[-2]
        px2, py2 = p_hist.positions[-1]
        vx1, vy1 = v_hist.positions[-2]
        vx2, vy2 = v_hist.positions[-1]

        dp = (px2 - px1, py2 - py1)
        dv = (vx2 - vx1, vy2 - vy1)

        dot = dp[0] * dv[0] + dp[1] * dv[1]
        mag_p = math.sqrt(dp[0] ** 2 + dp[1] ** 2)
        mag_v = math.sqrt(dv[0] ** 2 + dv[1] ** 2)

        if mag_p == 0 or mag_v == 0:
            return 0.0

        return dot / (mag_p * mag_v)

    def _compute_confidence(self, interaction: Interaction) -> float:
        """Weighted confidence based on duration, proximity, and velocity.

        The score is normalised to [0, 1].
        """
        # Duration component — saturates at 60 frames
        duration_score = min(interaction.duration / 60.0, 1.0)

        # Proximity component — closer is higher confidence
        proximity_score = max(
            0.0,
            1.0 - (interaction.current_distance / self.distance_threshold),
        )

        # Velocity component — higher closing speed raises confidence
        velocity_score = 0.0
        if interaction.relative_velocity < 0:
            velocity_score = min(abs(interaction.relative_velocity) / 10.0, 1.0)

        confidence = (
            self.confidence_base * proximity_score
            + 0.3 * duration_score
            + 0.2 * velocity_score
        )
        return round(min(confidence, 1.0), 4)

    @staticmethod
    def _build_motion_snapshot(
        person: Track, vehicle: Track, frame_number: int
    ) -> dict:
        """Build a per-frame motion snapshot dictionary."""
        return {
            "frame": frame_number,
            "person_speed": person.instantaneous_speed,
            "person_direction": person.direction,
            "vehicle_speed": vehicle.instantaneous_speed,
            "vehicle_direction": vehicle.direction,
            "person_center": person.center,
            "vehicle_center": vehicle.center,
        }

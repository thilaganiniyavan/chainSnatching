"""Unit tests for the InteractionManager.

Tests cover:
- Interaction creation from relationship data
- Lifecycle transitions (NEW -> ACTIVE -> LINGERING -> ENDED -> ARCHIVED)
- Distance / velocity / acceleration computation
- Multiple simultaneous interactions
- Confidence scoring
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.track import Track
from src.core.models.track_history import TrackHistory
from src.core.models.relationship import Relationship
from src.core.models.interaction import InteractionState
from src.behavior.interaction_manager import InteractionManager


# ======================================================================
# Helpers
# ======================================================================

def _make_track(
    tracking_id: int,
    class_name: str,
    center: tuple[int, int],
    speed: float = 0.0,
    direction: float = 0.0,
    positions: list[tuple[int, int]] | None = None,
) -> Track:
    """Create a Track with optional history for testing."""
    history = None
    if positions and len(positions) >= 1:
        history = TrackHistory(
            tracking_id=tracking_id,
            positions=positions,
        )
    track = Track(
        tracking_id=tracking_id,
        class_name=class_name,
        center=center,
        instantaneous_speed=speed,
        average_speed=speed,
        direction=direction,
        history=history,
    )
    return track


def _make_relationship(
    person_id: int,
    vehicle_id: int,
    distance: float,
    timestamp: float = 0.0,
) -> Relationship:
    return Relationship(
        subject_id=person_id,
        subject_class="person",
        object_id=vehicle_id,
        object_class="motorcycle",
        relationship_type="near",
        distance=distance,
        timestamp=timestamp,
    )


# ======================================================================
# Tests — Creation
# ======================================================================

class TestInteractionCreation:
    """Tests that interactions are correctly created."""

    def test_create_returns_interaction(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        interaction = manager.create(person, vehicle, frame_number=5, distance=50.0)

        assert interaction.person_track_id == 1
        assert interaction.vehicle_track_id == 2
        assert interaction.start_frame == 5
        assert interaction.state == InteractionState.NEW
        assert interaction.current_distance == 50.0

    def test_create_multiple(self):
        manager = InteractionManager()
        p1 = _make_track(1, "person", (100, 200))
        v1 = _make_track(2, "motorcycle", (130, 210))
        p2 = _make_track(3, "person", (300, 400))
        v2 = _make_track(4, "car", (330, 410))

        i1 = manager.create(p1, v1, frame_number=1, distance=40.0)
        i2 = manager.create(p2, v2, frame_number=1, distance=60.0)

        assert i1.interaction_id != i2.interaction_id
        assert len(manager.get_all()) == 2

    def test_unique_ids(self):
        manager = InteractionManager()
        ids = set()
        for i in range(20):
            p = _make_track(i * 2, "person", (i, i))
            v = _make_track(i * 2 + 1, "motorcycle", (i + 10, i + 10))
            interaction = manager.create(p, v, frame_number=1, distance=100.0)
            ids.add(interaction.interaction_id)
        assert len(ids) == 20


# ======================================================================
# Tests — Lifecycle Transitions
# ======================================================================

class TestLifecycleTransitions:
    """Tests that interactions transition through the correct lifecycle states."""

    def test_new_to_active(self):
        """An interaction should transition from NEW to ACTIVE on the second update."""
        manager = InteractionManager(linger_frames=5, end_frames=10)
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]

        # Frame 1 — create
        manager.update(rels, [person, vehicle], frame_number=1)
        interaction = manager.get_active()[0]
        assert interaction.state == InteractionState.NEW

        # Frame 2 — should transition to ACTIVE
        manager.update(rels, [person, vehicle], frame_number=2)
        assert interaction.state == InteractionState.ACTIVE

    def test_active_to_lingering(self):
        """Interaction transitions to LINGERING after linger_frames without proximity."""
        manager = InteractionManager(linger_frames=3, end_frames=5)
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]

        # Create and activate
        manager.update(rels, [person, vehicle], frame_number=1)
        manager.update(rels, [person, vehicle], frame_number=2)
        interaction = manager.get_all()[0]
        assert interaction.state == InteractionState.ACTIVE

        # Remove proximity for linger_frames
        for f in range(3, 3 + 3):
            manager.update([], [person, vehicle], f)

        assert interaction.state == InteractionState.LINGERING

    def test_lingering_to_ended(self):
        """Interaction transitions to ENDED after end_frames in LINGERING."""
        manager = InteractionManager(linger_frames=2, end_frames=3)
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]

        # Create and activate
        manager.update(rels, [person, vehicle], frame_number=1)
        manager.update(rels, [person, vehicle], frame_number=2)

        # Linger (2 frames) + End (3 frames) = 5 frames without proximity
        for f in range(3, 3 + 5):
            manager.update([], [person, vehicle], f)

        assert interaction_state(manager) == InteractionState.ENDED

    def test_reactivation_from_lingering(self):
        """An interaction in LINGERING should reactivate if proximity resumes."""
        manager = InteractionManager(linger_frames=2, end_frames=10)
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]

        # Create and activate
        manager.update(rels, [person, vehicle], frame_number=1)
        manager.update(rels, [person, vehicle], frame_number=2)

        # Go to LINGERING
        for f in range(3, 3 + 2):
            manager.update([], [person, vehicle], f)

        interaction = manager.get_all()[0]
        assert interaction.state == InteractionState.LINGERING

        # Re-appear
        manager.update(rels, [person, vehicle], frame_number=6)
        assert interaction.state == InteractionState.ACTIVE

    def test_manual_terminate(self):
        """terminate() should force interaction to ENDED."""
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        interaction = manager.create(person, vehicle, frame_number=1, distance=50.0)
        manager.terminate(interaction.interaction_id)
        assert interaction.state == InteractionState.ENDED


def interaction_state(manager: InteractionManager) -> InteractionState:
    """Get the state of the first (and assumed only) interaction."""
    all_ints = manager.get_all()
    assert len(all_ints) > 0
    return all_ints[0].state


# ======================================================================
# Tests — Distance / Velocity / Acceleration
# ======================================================================

class TestKinematics:
    """Tests spatial and kinematic computations."""

    def test_distance_tracking(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (200, 200))

        rels = [_make_relationship(1, 2, 100.0)]
        manager.update(rels, [person, vehicle], frame_number=1)

        # Closer distance
        rels2 = [_make_relationship(1, 2, 60.0)]
        manager.update(rels2, [person, vehicle], frame_number=2)

        interaction = manager.get_all()[0]
        assert interaction.min_distance == 60.0
        assert interaction.max_distance == 100.0
        assert interaction.current_distance == 60.0

    def test_relative_velocity(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (200, 200))

        rels1 = [_make_relationship(1, 2, 100.0)]
        manager.update(rels1, [person, vehicle], frame_number=1)

        rels2 = [_make_relationship(1, 2, 80.0)]
        manager.update(rels2, [person, vehicle], frame_number=2)

        interaction = manager.get_all()[0]
        # Velocity = 80 - 100 = -20 (closing)
        assert interaction.relative_velocity == -20.0

    def test_avg_distance(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (200, 200))

        distances = [100.0, 80.0, 60.0]
        for i, d in enumerate(distances):
            rels = [_make_relationship(1, 2, d)]
            manager.update(rels, [person, vehicle], frame_number=i + 1)

        interaction = manager.get_all()[0]
        expected_avg = sum(distances) / len(distances)
        assert abs(interaction.avg_distance - expected_avg) < 0.01


# ======================================================================
# Tests — Multiple Simultaneous Interactions
# ======================================================================

class TestMultipleInteractions:
    """Tests support for multiple simultaneous interactions."""

    def test_two_simultaneous_interactions(self):
        manager = InteractionManager()
        p1 = _make_track(1, "person", (100, 200))
        v1 = _make_track(2, "motorcycle", (130, 210))
        p2 = _make_track(3, "person", (400, 500))
        v2 = _make_track(4, "car", (420, 510))

        rels = [
            _make_relationship(1, 2, 40.0),
            _make_relationship(3, 4, 30.0),
        ]
        tracks = [p1, v1, p2, v2]

        manager.update(rels, tracks, frame_number=1)
        assert len(manager.get_all()) == 2

    def test_independent_lifecycle(self):
        """Two interactions should age independently."""
        manager = InteractionManager(linger_frames=2, end_frames=3)

        p1 = _make_track(1, "person", (100, 200))
        v1 = _make_track(2, "motorcycle", (130, 210))
        p2 = _make_track(3, "person", (400, 500))
        v2 = _make_track(4, "car", (420, 510))

        rel1 = _make_relationship(1, 2, 40.0)
        rel2 = _make_relationship(3, 4, 30.0)

        tracks = [p1, v1, p2, v2]

        # Both active
        manager.update([rel1, rel2], tracks, frame_number=1)
        manager.update([rel1, rel2], tracks, frame_number=2)

        # Only interaction 2 continues
        for f in range(3, 10):
            manager.update([rel2], tracks, f)

        all_ints = manager.get_all()
        states = {i.person_track_id: i.state for i in all_ints}

        assert states[1] in (InteractionState.ENDED, InteractionState.ARCHIVED)
        assert states[3] == InteractionState.ACTIVE


# ======================================================================
# Tests — Confidence Scoring
# ======================================================================

class TestConfidence:
    """Tests interaction confidence computation."""

    def test_confidence_increases_with_proximity(self):
        manager = InteractionManager(distance_threshold=150.0)
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (110, 210))

        # Very close interaction — need at least 2 updates so confidence is computed
        rels = [_make_relationship(1, 2, 10.0)]
        manager.update(rels, [person, vehicle], frame_number=1)
        manager.update(rels, [person, vehicle], frame_number=2)

        interaction = manager.get_all()[0]
        close_confidence = interaction.interaction_confidence

        # Farther interaction
        manager2 = InteractionManager(distance_threshold=150.0)
        rels_far = [_make_relationship(1, 2, 140.0)]
        manager2.update(rels_far, [person, vehicle], frame_number=1)
        manager2.update(rels_far, [person, vehicle], frame_number=2)
        far_confidence = manager2.get_all()[0].interaction_confidence

        assert close_confidence > far_confidence

    def test_confidence_bounded(self):
        """Confidence should be in [0, 1]."""
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (105, 205))

        rels = [_make_relationship(1, 2, 5.0)]
        for f in range(1, 100):
            manager.update(rels, [person, vehicle], f)

        interaction = manager.get_all()[0]
        assert 0.0 <= interaction.interaction_confidence <= 1.0


# ======================================================================
# Tests — Query APIs
# ======================================================================

class TestQueryAPIs:
    """Tests for get_active, get_completed, get_by_state."""

    def test_get_active(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]
        manager.update(rels, [person, vehicle], frame_number=1)

        active = manager.get_active()
        assert len(active) == 1
        assert active[0].state == InteractionState.NEW

    def test_get_completed_empty(self):
        manager = InteractionManager()
        assert manager.get_completed() == []

    def test_get_by_state(self):
        manager = InteractionManager()
        person = _make_track(1, "person", (100, 200))
        vehicle = _make_track(2, "motorcycle", (130, 210))

        rels = [_make_relationship(1, 2, 50.0)]
        manager.update(rels, [person, vehicle], frame_number=1)

        new_ints = manager.get_by_state(InteractionState.NEW)
        assert len(new_ints) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

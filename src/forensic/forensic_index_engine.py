"""Forensic Index Engine — multi-attribute inverted search indexing.

Maintains fast O(1) inverted indices across:
- event_id, video_id, decision, signature_template
- person_track_id, vehicle_track_id
- behaviour_patterns (APPROACH, INTERACTION, ESCAPE)
- detected_actions (Reaching, Grabbing, Pulling)
- tags & keyword tokens

Provides range lookups for timestamps and confidence score thresholds.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Set

from src.core.models.forensic_event import ForensicEvent


class ForensicIndexEngine:
    """Multi-attribute search index engine for ForensicEvent records."""

    def __init__(self) -> None:
        # Inverted index mappings: key -> set of event_ids
        self._by_event_id: dict[str, ForensicEvent] = {}
        self._by_video_id: dict[str, Set[str]] = defaultdict(set)
        self._by_decision: dict[str, Set[str]] = defaultdict(set)
        self._by_signature_name: dict[str, Set[str]] = defaultdict(set)
        self._by_track_id: dict[int, Set[str]] = defaultdict(set)
        self._by_pattern: dict[str, Set[str]] = defaultdict(set)
        self._by_action: dict[str, Set[str]] = defaultdict(set)
        self._by_tag: dict[str, Set[str]] = defaultdict(set)

    def index_event(self, event: ForensicEvent) -> None:
        """Add or update an event in all inverted indices."""
        eid = event.event_id
        self._by_event_id[eid] = event

        self._by_video_id[event.video_id].add(eid)
        self._by_decision[event.decision].add(eid)
        self._by_signature_name[event.matched_signature_name].add(eid)

        if event.person_track_id != -1:
            self._by_track_id[event.person_track_id].add(eid)
        if event.vehicle_track_id != -1:
            self._by_track_id[event.vehicle_track_id].add(eid)

        for pat in event.behaviour_patterns:
            self._by_pattern[pat].add(eid)

        for act in event.detected_actions:
            self._by_action[act].add(eid)

        for tag in event.tags:
            self._by_tag[tag.lower().strip()].add(eid)

    def remove_event(self, event_id: str) -> bool:
        """Remove an event from all inverted indices."""
        event = self._by_event_id.pop(event_id, None)
        if event is None:
            return False

        self._by_video_id[event.video_id].discard(event_id)
        self._by_decision[event.decision].discard(event_id)
        self._by_signature_name[event.matched_signature_name].discard(event_id)
        self._by_track_id[event.person_track_id].discard(event_id)
        self._by_track_id[event.vehicle_track_id].discard(event_id)

        for pat in event.behaviour_patterns:
            self._by_pattern[pat].discard(event_id)
        for act in event.detected_actions:
            self._by_action[act].discard(event_id)
        for tag in event.tags:
            self._by_tag[tag.lower().strip()].discard(event_id)

        return True

    def get_event(self, event_id: str) -> ForensicEvent | None:
        """Get event record by event_id."""
        return self._by_event_id.get(event_id)

    def get_all_events(self) -> list[ForensicEvent]:
        """Get list of all indexed ForensicEvents."""
        return list(self._by_event_id.values())

    def lookup(
        self,
        video_id: str | None = None,
        decision: str | None = None,
        track_id: int | None = None,
        pattern: str | None = None,
        action: str | None = None,
        signature_name: str | None = None,
        tag: str | None = None,
    ) -> Set[str]:
        """Intersect inverted indices for specified query filters and return matching event_ids."""
        candidate_sets: list[Set[str]] = []

        if video_id is not None:
            candidate_sets.append(self._by_video_id.get(video_id, set()))
        if decision is not None:
            candidate_sets.append(self._by_decision.get(decision, set()))
        if track_id is not None:
            candidate_sets.append(self._by_track_id.get(track_id, set()))
        if pattern is not None:
            candidate_sets.append(self._by_pattern.get(pattern, set()))
        if action is not None:
            candidate_sets.append(self._by_action.get(action, set()))
        if signature_name is not None:
            candidate_sets.append(self._by_signature_name.get(signature_name, set()))
        if tag is not None:
            candidate_sets.append(self._by_tag.get(tag.lower().strip(), set()))

        if not candidate_sets:
            return set(self._by_event_id.keys())

        # Set intersection across criteria
        result_set = candidate_sets[0].copy()
        for s in candidate_sets[1:]:
            result_set.intersection_update(s)

        return result_set

    def clear(self) -> None:
        """Clear all inverted indices."""
        self._by_event_id.clear()
        self._by_video_id.clear()
        self._by_decision.clear()
        self._by_signature_name.clear()
        self._by_track_id.clear()
        self._by_pattern.clear()
        self._by_action.clear()
        self._by_tag.clear()

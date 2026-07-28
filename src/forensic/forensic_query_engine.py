"""Forensic Query Engine — expressive search and filtering APIs.

Provides investigator query methods:
- add_event()
- update_event()
- delete_event()
- get_event()
- search_events()
- filter_events()
- export_events()

Supports querying by decision label, track ID, pattern, action, confidence, score, and timestamps.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from src.core.models.forensic_event import ForensicEvent
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.forensic.forensic_index_engine import ForensicIndexEngine


class ForensicQueryEngine:
    """Investigator Query Engine providing search, filtering, and event record management."""

    def __init__(self) -> None:
        self.index_engine = ForensicIndexEngine()

    def create_event_from_signature(
        self,
        sig_result: SnatchSignatureResult,
        video_id: str = "default_video",
        location: str = "Camera 1",
    ) -> ForensicEvent:
        """Construct a ForensicEvent record from a SnatchSignatureResult and index it."""
        event_id = f"EVT-{sig_result.signature_id}"

        # Extract behaviour patterns and detected actions
        patterns = sig_result.behaviour_evidence
        actions = sig_result.action_evidence

        # Build tags
        tags = [sig_result.decision, sig_result.matched_signature_name]
        tags.extend(patterns)
        tags.extend(actions)

        # Build evidence summary
        summary = {
            "matched_count": len(sig_result.matched_evidence),
            "missing_count": len(sig_result.missing_evidence),
        }

        # Extract track IDs from metadata
        person_tr = sig_result.metadata.get("person_track_id", -1)
        vehicle_tr = sig_result.metadata.get("vehicle_track_id", -1)

        event = ForensicEvent(
            event_id=event_id,
            video_id=video_id,
            interaction_id=sig_result.interaction_id,
            fusion_id=sig_result.fusion_id,
            signature_id=sig_result.signature_id,
            timestamp=sig_result.metadata.get("timestamp", time.time()),
            start_frame=sig_result.metadata.get("start_frame", 0),
            end_frame=sig_result.metadata.get("end_frame", 0),
            duration_seconds=sig_result.metadata.get("duration_seconds", 0.0),
            person_track_id=person_tr,
            vehicle_track_id=vehicle_tr,
            location=location,
            decision=sig_result.decision,
            signature_score=sig_result.signature_score,
            confidence=sig_result.confidence,
            matched_signature_name=sig_result.matched_signature_name,
            behaviour_patterns=patterns,
            detected_actions=actions,
            evidence_summary=summary,
            behaviour_graph_ref=f"outputs/behaviour_graph.json#{sig_result.interaction_id}",
            action_timeline_ref=f"outputs/action_results.json#{sig_result.interaction_id}",
            roi_ref=f"outputs/interaction_rois.json#{sig_result.interaction_id}",
            pose_ref=f"outputs/pose_results.json#{sig_result.interaction_id}",
            skeleton_ref=f"outputs/skeleton_sequences.json#{sig_result.interaction_id}",
            fusion_ref=f"outputs/fused_interactions.json#{sig_result.fusion_id}",
            thumbnail_path=f"outputs/forensic_thumbnails/thumb_{event_id}.jpg",
            video_clip_path=f"outputs/forensic_clips/event_{event_id}.avi",
            investigator_notes=sig_result.recommendation,
            tags=list(set(tags)),
        )

        self.index_engine.index_event(event)
        return event

    def add_event(self, event: ForensicEvent) -> str:
        """Index a ForensicEvent record and return its event_id."""
        self.index_engine.index_event(event)
        return event.event_id

    def update_event(self, event_id: str, updates: dict[str, Any]) -> Optional[ForensicEvent]:
        """Update fields of an existing ForensicEvent record."""
        event = self.index_engine.get_event(event_id)
        if event is None:
            return None

        for k, v in updates.items():
            if hasattr(event, k):
                setattr(event, k, v)

        self.index_engine.index_event(event)
        return event

    def delete_event(self, event_id: str) -> bool:
        """Delete a ForensicEvent record by event_id."""
        return self.index_engine.remove_event(event_id)

    def get_event(self, event_id: str) -> Optional[ForensicEvent]:
        """Retrieve a ForensicEvent by event_id."""
        return self.index_engine.get_event(event_id)

    def search_events(self, query_string: str) -> list[ForensicEvent]:
        """Search events matching query string tokens against decisions, patterns, actions, and tags."""
        tokens = [t.lower().strip() for t in query_string.split() if t.strip()]
        if not tokens:
            return self.index_engine.get_all_events()

        all_events = self.index_engine.get_all_events()
        matched: list[ForensicEvent] = []

        for e in all_events:
            text_blob = f"{e.event_id} {e.decision} {e.matched_signature_name} {' '.join(e.behaviour_patterns)} {' '.join(e.detected_actions)} {' '.join(e.tags)} {e.investigator_notes}".lower()
            if all(tok in text_blob for tok in tokens):
                matched.append(e)

        return matched

    def filter_events(
        self,
        decision: str | None = None,
        min_confidence: float | None = None,
        min_score: float | None = None,
        track_id: int | None = None,
        pattern: str | None = None,
        action: str | None = None,
        video_id: str | None = None,
        tag: str | None = None,
    ) -> list[ForensicEvent]:
        """Filter ForensicEvents by decision label, minimum confidence, minimum score, track_id, pattern, action, or tag."""
        candidate_ids = self.index_engine.lookup(
            video_id=video_id,
            decision=decision,
            track_id=track_id,
            pattern=pattern,
            action=action,
            tag=tag,
        )

        filtered: list[ForensicEvent] = []
        for eid in candidate_ids:
            e = self.index_engine.get_event(eid)
            if e is None:
                continue

            if min_confidence is not None and e.confidence < min_confidence:
                continue
            if min_score is not None and e.signature_score < min_score:
                continue

            filtered.append(e)

        return sorted(filtered, key=lambda x: x.signature_score, reverse=True)

    def get_all_events(self) -> list[ForensicEvent]:
        """Return all indexed ForensicEvent records."""
        return self.index_engine.get_all_events()

    def clear(self) -> None:
        """Clear all indexed events."""
        self.index_engine.clear()

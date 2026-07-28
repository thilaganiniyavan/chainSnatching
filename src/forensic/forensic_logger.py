"""Forensic Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``forensic_events.json``
- ``forensic_index.csv``
- ``retrieval_statistics.csv``
- ``forensic_index_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.forensic_event import ForensicEvent


class ForensicLogger:
    """Logs and exports ForensicEvents to JSON, CSV, and markdown index reports."""

    def __init__(self) -> None:
        self._events: list[ForensicEvent] = []

    def log_event(self, event: ForensicEvent) -> None:
        """Store a ForensicEvent instance for export."""
        self._events.append(event)

    def log_events(self, events: list[ForensicEvent]) -> None:
        """Store multiple ForensicEvent instances for export."""
        for e in events:
            self.log_event(e)

    def export_json(self, output_path: str) -> None:
        """Export all logged forensic events to forensic_events.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        event_dicts = [self._event_to_dict(e) for e in self._events]

        avg_score = (
            sum(e.signature_score for e in self._events) / max(1, len(self._events))
        )
        avg_conf = (
            sum(e.confidence for e in self._events) / max(1, len(self._events))
        )

        payload = {
            "forensic_events": event_dicts,
            "summary": {
                "total_indexed_events": len(self._events),
                "average_signature_score": round(avg_score, 4),
                "average_confidence": round(avg_conf, 4),
                "decision_counts": self._count_decisions(),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged forensic events to forensic_index.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "event_id",
            "video_id",
            "interaction_id",
            "fusion_id",
            "signature_id",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "person_track_id",
            "vehicle_track_id",
            "location",
            "decision",
            "signature_score",
            "confidence",
            "matched_signature_name",
            "behaviour_patterns",
            "detected_actions",
            "thumbnail_path",
            "video_clip_path",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for e in self._events:
                writer.writerow(
                    {
                        "event_id": e.event_id,
                        "video_id": e.video_id,
                        "interaction_id": e.interaction_id,
                        "fusion_id": e.fusion_id,
                        "signature_id": e.signature_id,
                        "start_frame": e.start_frame,
                        "end_frame": e.end_frame,
                        "duration_seconds": e.duration_seconds,
                        "person_track_id": e.person_track_id,
                        "vehicle_track_id": e.vehicle_track_id,
                        "location": e.location,
                        "decision": e.decision,
                        "signature_score": e.signature_score,
                        "confidence": e.confidence,
                        "matched_signature_name": e.matched_signature_name,
                        "behaviour_patterns": " -> ".join(e.behaviour_patterns),
                        "detected_actions": ", ".join(e.detected_actions),
                        "thumbnail_path": e.thumbnail_path,
                        "video_clip_path": e.video_clip_path,
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted forensic_index_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        avg_score = (
            sum(e.signature_score for e in self._events) / max(1, len(self._events))
        )
        counts = self._count_decisions()

        lines: list[str] = []
        lines.append("# Forensic Indexing & Retrieval Engine — Index Report\n")
        lines.append(f"**Total Indexed Forensic Events:** {len(self._events)}")
        lines.append(f"**Average Signature Match Score:** {avg_score:.2f}\n")

        lines.append("## Indexed Decisions Summary\n")
        lines.append("| Decision Label | Event Count | Percentage (%) |")
        lines.append("|---|---|---|")

        total = max(1, len(self._events))
        for dec_label, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {dec_label} | {cnt} | {(cnt / total) * 100.0:.1f}% |")

        lines.append("\n## Searchable Forensic Event Registry\n")
        lines.append("| Event ID | Video ID | Location | Decision | Score | Confidence | Patterns | Actions | Notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for e in self._events:
            pat_str = ", ".join(e.behaviour_patterns) if e.behaviour_patterns else "—"
            act_str = ", ".join(e.detected_actions) if e.detected_actions else "—"
            lines.append(
                f"| **{e.event_id}** | {e.video_id} | {e.location} | **{e.decision}** | "
                f"{e.signature_score:.2f} | {e.confidence:.0%} | {pat_str} | {act_str} | {e.investigator_notes[:50]} |"
            )

        lines.append("\n---\n*Report generated by the Forensic Indexing & Retrieval Engine.*\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_all(
        self,
        json_path: str,
        csv_path: str,
        report_path: str,
    ) -> None:
        """Export JSON, CSV, and markdown quality report."""
        self.export_json(json_path)
        self.export_csv(csv_path)
        self.export_quality_report(report_path)

    def get_events(self) -> list[ForensicEvent]:
        """Return all logged forensic events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear internal log storage."""
        self._events.clear()

    def _count_decisions(self) -> dict[str, int]:
        """Count occurrences per decision label."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.decision] = counts.get(e.decision, 0) + 1
        return counts

    @staticmethod
    def _event_to_dict(event: ForensicEvent) -> dict[str, Any]:
        """Serialise a ForensicEvent instance to a dictionary."""
        return {
            "event_id": event.event_id,
            "video_id": event.video_id,
            "interaction_id": event.interaction_id,
            "fusion_id": event.fusion_id,
            "signature_id": event.signature_id,
            "timestamp": event.timestamp,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "duration_seconds": event.duration_seconds,
            "person_track_id": event.person_track_id,
            "vehicle_track_id": event.vehicle_track_id,
            "location": event.location,
            "decision": event.decision,
            "signature_score": event.signature_score,
            "confidence": event.confidence,
            "matched_signature_name": event.matched_signature_name,
            "behaviour_patterns": event.behaviour_patterns,
            "detected_actions": event.detected_actions,
            "evidence_summary": event.evidence_summary,
            "behaviour_graph_ref": event.behaviour_graph_ref,
            "action_timeline_ref": event.action_timeline_ref,
            "roi_ref": event.roi_ref,
            "pose_ref": event.pose_ref,
            "skeleton_ref": event.skeleton_ref,
            "fusion_ref": event.fusion_ref,
            "thumbnail_path": event.thumbnail_path,
            "video_clip_path": event.video_clip_path,
            "investigator_notes": event.investigator_notes,
            "tags": event.tags,
            "metadata": event.metadata,
        }

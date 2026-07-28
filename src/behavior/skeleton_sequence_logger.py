"""Skeleton Sequence Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``skeleton_sequences.json``
- ``sequence_statistics.csv``
- ``sequence_quality_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.skeleton_sequence import SkeletonSequence


class SkeletonSequenceLogger:
    """Logs and exports SkeletonSequences to JSON, CSV, and markdown quality reports."""

    def __init__(self) -> None:
        self._sequences: list[SkeletonSequence] = []

    def log_sequence(self, sequence: SkeletonSequence) -> None:
        """Store a SkeletonSequence instance for export."""
        self._sequences.append(sequence)

    def log_sequences(self, sequences: list[SkeletonSequence]) -> None:
        """Store multiple SkeletonSequence instances for export."""
        for s in sequences:
            self.log_sequence(s)

    def export_json(self, output_path: str, include_raw_tensors: bool = False) -> None:
        """Export all logged sequences to skeleton_sequences.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        seq_dicts = [
            self._sequence_to_dict(s, include_raw_tensors=include_raw_tensors)
            for s in self._sequences
        ]

        accepted_cnt = sum(1 for s in self._sequences if s.is_accepted)
        rejected_cnt = len(self._sequences) - accepted_cnt

        payload = {
            "sequences": seq_dicts,
            "summary": {
                "total_sequences": len(self._sequences),
                "accepted_sequences": accepted_cnt,
                "rejected_sequences": rejected_cnt,
                "acceptance_rate_pct": round((accepted_cnt / max(1, len(self._sequences))) * 100.0, 2),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged sequences to sequence_statistics.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "sequence_id",
            "interaction_id",
            "person_track_id",
            "start_frame",
            "end_frame",
            "frame_count",
            "duration_seconds",
            "topology",
            "num_joints",
            "normalization_method",
            "quality_score",
            "completeness_score",
            "is_accepted",
            "rejection_reason",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for s in self._sequences:
                writer.writerow(
                    {
                        "sequence_id": s.sequence_id,
                        "interaction_id": s.interaction_id,
                        "person_track_id": s.person_track_id,
                        "start_frame": s.start_frame,
                        "end_frame": s.end_frame,
                        "frame_count": s.frame_count,
                        "duration_seconds": s.duration_seconds,
                        "topology": s.topology,
                        "num_joints": s.num_joints,
                        "normalization_method": s.normalization_method,
                        "quality_score": s.quality_score,
                        "completeness_score": s.completeness_score,
                        "is_accepted": s.is_accepted,
                        "rejection_reason": s.rejection_reason,
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted sequence_quality_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        accepted_cnt = sum(1 for s in self._sequences if s.is_accepted)
        rejected_cnt = len(self._sequences) - accepted_cnt

        lines: list[str] = []
        lines.append("# Skeleton Sequence Builder — Quality Evaluation Report\n")
        lines.append(f"**Total Sequences Built:** {len(self._sequences)}")
        lines.append(f"**Accepted Sequences:** {accepted_cnt}")
        lines.append(f"**Rejected Sequences:** {rejected_cnt}")
        lines.append(f"**Acceptance Rate:** {(accepted_cnt / max(1, len(self._sequences))) * 100.0:.1f}%\n")

        lines.append("## Sequence Details\n")
        lines.append("| Sequence ID | Interaction ID | Track ID | Frames | Normalization | Quality | Status | Rejection Reason |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for s in self._sequences:
            status = "ACCEPTED" if s.is_accepted else "REJECTED"
            lines.append(
                f"| {s.sequence_id} | {s.interaction_id} | {s.person_track_id} | "
                f"{s.frame_count} ({s.duration_seconds:.1f}s) | {s.normalization_method} | "
                f"{s.quality_score:.2f} | {status} | {s.rejection_reason} |"
            )

        lines.append("\n---\n*Report generated by the Skeleton Sequence Builder Engine.*\n")

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

    def get_sequences(self) -> list[SkeletonSequence]:
        """Return all logged sequences."""
        return list(self._sequences)

    def clear(self) -> None:
        """Clear internal sequence log storage."""
        self._sequences.clear()

    @staticmethod
    def _sequence_to_dict(
        sequence: SkeletonSequence,
        include_raw_tensors: bool = False,
    ) -> dict[str, Any]:
        """Serialise a SkeletonSequence instance to a dictionary."""
        d = {
            "sequence_id": sequence.sequence_id,
            "interaction_id": sequence.interaction_id,
            "person_track_id": sequence.person_track_id,
            "start_frame": sequence.start_frame,
            "end_frame": sequence.end_frame,
            "frame_count": sequence.frame_count,
            "duration_seconds": sequence.duration_seconds,
            "topology": sequence.topology,
            "num_joints": sequence.num_joints,
            "normalization_method": sequence.normalization_method,
            "quality_score": sequence.quality_score,
            "completeness_score": sequence.completeness_score,
            "is_accepted": sequence.is_accepted,
            "rejection_reason": sequence.rejection_reason,
            "frame_indices": sequence.frame_indices,
            "timestamps": sequence.timestamps,
            "metadata": sequence.metadata,
        }

        if include_raw_tensors:
            d["skeleton_tensor"] = sequence.skeleton_tensor.tolist()
            d["joint_confidence_matrix"] = sequence.joint_confidence_matrix.tolist()

        return d

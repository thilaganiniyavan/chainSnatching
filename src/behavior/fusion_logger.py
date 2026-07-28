"""Fusion Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``fused_interactions.json``
- ``fusion_statistics.csv``
- ``fusion_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.fused_interaction import FusedInteraction


class FusionLogger:
    """Logs and exports FusedInteractions to JSON, CSV, and markdown quality reports."""

    def __init__(self) -> None:
        self._fusions: list[FusedInteraction] = []

    def log_fusion(self, fusion: FusedInteraction) -> None:
        """Store a FusedInteraction instance for export."""
        self._fusions.append(fusion)

    def log_fusions(self, fusions: list[FusedInteraction]) -> None:
        """Store multiple FusedInteraction instances for export."""
        for f in fusions:
            self.log_fusion(f)

    def export_json(self, output_path: str) -> None:
        """Export all logged fused interactions to fused_interactions.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fusion_dicts = [self._fusion_to_dict(f) for f in self._fusions]

        avg_conf = (
            sum(f.fusion_confidence for f in self._fusions) / max(1, len(self._fusions))
        )

        payload = {
            "fused_interactions": fusion_dicts,
            "summary": {
                "total_fused_interactions": len(self._fusions),
                "average_fusion_confidence": round(avg_conf, 4),
                "fusion_strategies_used": list(set(f.fusion_strategy for f in self._fusions)),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged fused interactions to fusion_statistics.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "fusion_id",
            "interaction_id",
            "person_track_id",
            "vehicle_track_id",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "behaviour_patterns",
            "behaviour_confidence",
            "action_confidence",
            "fusion_confidence",
            "fusion_strategy",
            "explanation_text",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for f in self._fusions:
                writer.writerow(
                    {
                        "fusion_id": f.fusion_id,
                        "interaction_id": f.interaction_id,
                        "person_track_id": f.person_track_id,
                        "vehicle_track_id": f.vehicle_track_id,
                        "start_frame": f.start_frame,
                        "end_frame": f.end_frame,
                        "duration_seconds": f.duration_seconds,
                        "behaviour_patterns": " -> ".join(f.behaviour_patterns),
                        "behaviour_confidence": f.behaviour_confidence,
                        "action_confidence": f.action_confidence,
                        "fusion_confidence": f.fusion_confidence,
                        "fusion_strategy": f.fusion_strategy,
                        "explanation_text": f.explanation_text,
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted fusion_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        avg_conf = (
            sum(f.fusion_confidence for f in self._fusions) / max(1, len(self._fusions))
        )

        lines: list[str] = []
        lines.append("# Behaviour Fusion Engine — Research Report\n")
        lines.append(f"**Total Fused Interactions:** {len(self._fusions)}")
        lines.append(f"**Average Fusion Confidence:** {avg_conf:.0%}\n")

        lines.append("## Fused Interaction Details & Evidence Provenance\n")
        lines.append("| Fusion ID | Interaction ID | Patterns | Actions | Graph Conf | Action Conf | Fusion Conf | Strategy |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for f in self._fusions:
            pat_str = ", ".join(f.behaviour_patterns) if f.behaviour_patterns else "—"
            act_str = ", ".join(set(a["action_label"] for a in f.action_timeline if a["action_label"] != "Unknown")) or "—"
            lines.append(
                f"| {f.fusion_id} | {f.interaction_id} | {pat_str} | {act_str} | "
                f"{f.behaviour_confidence:.0%} | {f.action_confidence:.0%} | **{f.fusion_confidence:.0%}** | {f.fusion_strategy} |"
            )

        lines.append("\n## Explainable Forensic Statements\n")
        for f in self._fusions:
            lines.append(f"- **{f.fusion_id}**: *\"{f.explanation_text}\"*")

        lines.append("\n---\n*Report generated by the Behaviour Fusion Engine.*\n")

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

    def get_fusions(self) -> list[FusedInteraction]:
        """Return all logged fused interactions."""
        return list(self._fusions)

    def clear(self) -> None:
        """Clear internal log storage."""
        self._fusions.clear()

    @staticmethod
    def _fusion_to_dict(fusion: FusedInteraction) -> dict[str, Any]:
        """Serialise a FusedInteraction instance to a clean dictionary."""
        return {
            "fusion_id": fusion.fusion_id,
            "interaction_id": fusion.interaction_id,
            "person_track_id": fusion.person_track_id,
            "vehicle_track_id": fusion.vehicle_track_id,
            "start_frame": fusion.start_frame,
            "end_frame": fusion.end_frame,
            "duration_seconds": fusion.duration_seconds,
            "behaviour_patterns": fusion.behaviour_patterns,
            "action_timeline": fusion.action_timeline,
            "motion_evidence": fusion.motion_evidence,
            "spatial_evidence": fusion.spatial_evidence,
            "temporal_evidence": fusion.temporal_evidence,
            "action_evidence": fusion.action_evidence,
            "behaviour_confidence": fusion.behaviour_confidence,
            "action_confidence": fusion.action_confidence,
            "fusion_confidence": fusion.fusion_confidence,
            "fusion_strategy": fusion.fusion_strategy,
            "evidence_timeline": fusion.evidence_timeline,
            "explanation_text": fusion.explanation_text,
            "metadata": fusion.metadata,
        }

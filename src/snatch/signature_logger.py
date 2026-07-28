"""Signature Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``snatch_signature_results.json``
- ``signature_statistics.csv``
- ``signature_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.snatch_signature_result import SnatchSignatureResult


class SignatureLogger:
    """Logs and exports SnatchSignatureResults to JSON, CSV, and markdown quality reports."""

    def __init__(self) -> None:
        self._results: list[SnatchSignatureResult] = []

    def log_result(self, result: SnatchSignatureResult) -> None:
        """Store a SnatchSignatureResult instance for export."""
        self._results.append(result)

    def log_results(self, results: list[SnatchSignatureResult]) -> None:
        """Store multiple SnatchSignatureResult instances for export."""
        for r in results:
            self.log_result(r)

    def export_json(self, output_path: str) -> None:
        """Export all logged signature results to snatch_signature_results.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        sig_dicts = [self._result_to_dict(r) for r in self._results]

        avg_score = (
            sum(r.signature_score for r in self._results) / max(1, len(self._results))
        )
        flagged_count = sum(1 for r in self._results if r.decision in ("High Confidence Match", "Strong Match"))

        payload = {
            "snatch_signature_results": sig_dicts,
            "summary": {
                "total_evaluated_interactions": len(self._results),
                "average_signature_score": round(avg_score, 4),
                "flagged_high_confidence_matches": flagged_count,
                "decision_counts": self._count_decisions(),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged signature results to signature_statistics.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "signature_id",
            "interaction_id",
            "fusion_id",
            "matched_signature_name",
            "signature_score",
            "decision",
            "confidence",
            "matched_components_count",
            "missing_components_count",
            "recommendation",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self._results:
                writer.writerow(
                    {
                        "signature_id": r.signature_id,
                        "interaction_id": r.interaction_id,
                        "fusion_id": r.fusion_id,
                        "matched_signature_name": r.matched_signature_name,
                        "signature_score": r.signature_score,
                        "decision": r.decision,
                        "confidence": r.confidence,
                        "matched_components_count": len(r.matched_evidence),
                        "missing_components_count": len(r.missing_evidence),
                        "recommendation": r.recommendation,
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted signature_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        avg_score = (
            sum(r.signature_score for r in self._results) / max(1, len(self._results))
        )
        counts = self._count_decisions()

        lines: list[str] = []
        lines.append("# Snatch Signature Engine — Forensic Evaluation Report\n")
        lines.append(f"**Total Evaluated Interactions:** {len(self._results)}")
        lines.append(f"**Average Signature Match Score:** {avg_score:.2f}\n")

        lines.append("## Decision Distribution Summary\n")
        lines.append("| Decision Label | Interaction Count | Percentage (%) |")
        lines.append("|---|---|---|")

        total = max(1, len(self._results))
        for dec_label, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {dec_label} | {cnt} | {(cnt / total) * 100.0:.1f}% |")

        lines.append("\n## Detailed Forensic Signature Evaluations\n")
        lines.append("| Signature ID | Interaction ID | Score | Decision | Matched / Missing | Recommendation |")
        lines.append("|---|---|---|---|---|---|")

        for r in self._results:
            lines.append(
                f"| {r.signature_id} | {r.interaction_id} | **{r.signature_score:.2f}** | "
                f"**{r.decision}** | ✓ {len(r.matched_evidence)} / ✗ {len(r.missing_evidence)} | {r.recommendation} |"
            )

        lines.append("\n## Detailed Forensic Evidence Provenance\n")
        for r in self._results:
            lines.append(f"### Signature {r.signature_id} ({r.interaction_id})\n")
            lines.append("```text")
            lines.append(r.explanation_text)
            lines.append("```\n")

        lines.append("---\n*Report generated by the Snatch Signature Engine.*\n")

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

    def get_results(self) -> list[SnatchSignatureResult]:
        """Return all logged signature results."""
        return list(self._results)

    def clear(self) -> None:
        """Clear internal log storage."""
        self._results.clear()

    def _count_decisions(self) -> dict[str, int]:
        """Count occurrences per decision label."""
        counts: dict[str, int] = {}
        for r in self._results:
            counts[r.decision] = counts.get(r.decision, 0) + 1
        return counts

    @staticmethod
    def _result_to_dict(result: SnatchSignatureResult) -> dict[str, Any]:
        """Serialise a SnatchSignatureResult instance to a dictionary."""
        return {
            "signature_id": result.signature_id,
            "interaction_id": result.interaction_id,
            "fusion_id": result.fusion_id,
            "matched_signature_name": result.matched_signature_name,
            "signature_score": result.signature_score,
            "decision": result.decision,
            "confidence": result.confidence,
            "matched_evidence": result.matched_evidence,
            "missing_evidence": result.missing_evidence,
            "behaviour_evidence": result.behaviour_evidence,
            "action_evidence": result.action_evidence,
            "motion_evidence": result.motion_evidence,
            "spatial_evidence": result.spatial_evidence,
            "temporal_evidence": result.temporal_evidence,
            "evidence_timeline": result.evidence_timeline,
            "explanation_text": result.explanation_text,
            "recommendation": result.recommendation,
            "metadata": result.metadata,
        }

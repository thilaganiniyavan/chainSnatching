"""Action Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``action_results.json``
- ``action_statistics.csv``
- ``action_recognition_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.action_result import ActionResult


class ActionLogger:
    """Logs and exports ActionResults to JSON, CSV, and markdown quality reports."""

    def __init__(self) -> None:
        self._results: list[ActionResult] = []

    def log_result(self, result: ActionResult) -> None:
        """Store an ActionResult instance for export."""
        self._results.append(result)

    def log_results(self, results: list[ActionResult]) -> None:
        """Store multiple ActionResult instances for export."""
        for r in results:
            self.log_result(r)

    def export_json(self, output_path: str) -> None:
        """Export all logged action results to action_results.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        res_dicts = [self._result_to_dict(r) for r in self._results]

        avg_conf = (
            sum(r.action_confidence for r in self._results) / max(1, len(self._results))
        )
        avg_time = (
            sum(r.inference_time_ms for r in self._results) / max(1, len(self._results))
        )

        payload = {
            "action_results": res_dicts,
            "summary": {
                "total_action_classifications": len(self._results),
                "average_confidence": round(avg_conf, 4),
                "average_inference_time_ms": round(avg_time, 2),
                "models_used": list(set(r.model_name for r in self._results)),
                "action_counts": self._count_actions(),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged action results to action_statistics.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "action_id",
            "sequence_id",
            "interaction_id",
            "track_id",
            "predicted_action",
            "action_confidence",
            "inference_time_ms",
            "model_name",
            "model_version",
            "device_used",
            "skeleton_quality",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self._results:
                writer.writerow(
                    {
                        "action_id": r.action_id,
                        "sequence_id": r.sequence_id,
                        "interaction_id": r.interaction_id,
                        "track_id": r.track_id,
                        "predicted_action": r.predicted_action,
                        "action_confidence": r.action_confidence,
                        "inference_time_ms": r.inference_time_ms,
                        "model_name": r.model_name,
                        "model_version": r.model_version,
                        "device_used": r.device_used,
                        "skeleton_quality": r.skeleton_quality,
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted action_recognition_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        avg_conf = (
            sum(r.action_confidence for r in self._results) / max(1, len(self._results))
        )
        avg_time = (
            sum(r.inference_time_ms for r in self._results) / max(1, len(self._results))
        )
        counts = self._count_actions()

        lines: list[str] = []
        lines.append("# Human Action Recognition — Research Evaluation Report\n")
        lines.append(f"**Total Action Classifications:** {len(self._results)}")
        lines.append(f"**Average Prediction Confidence:** {avg_conf:.0%}")
        lines.append(f"**Average Model Inference Latency:** {avg_time:.2f} ms/sequence\n")

        lines.append("## Action Class Frequency Distribution\n")
        lines.append("| Action Class | Classification Count | Percentage (%) |")
        lines.append("|---|---|---|")

        total = max(1, len(self._results))
        for cls_name, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {cls_name} | {cnt} | {(cnt / total) * 100.0:.1f}% |")

        lines.append("\n## Detailed Predictions\n")
        lines.append("| Action ID | Sequence ID | Track ID | Predicted Action | Confidence | Model | Latency (ms) |")
        lines.append("|---|---|---|---|---|---|---|")

        for r in self._results[:20]:
            lines.append(
                f"| {r.action_id} | {r.sequence_id} | {r.track_id} | "
                f"**{r.predicted_action}** | {r.action_confidence:.0%} | {r.model_name} | {r.inference_time_ms:.1f}ms |"
            )

        lines.append("\n---\n*Report generated by the Human Action Recognition Framework.*\n")

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

    def get_results(self) -> list[ActionResult]:
        """Return all logged action results."""
        return list(self._results)

    def clear(self) -> None:
        """Clear internal log storage."""
        self._results.clear()

    def _count_actions(self) -> dict[str, int]:
        """Count occurrences per action class."""
        counts: dict[str, int] = {}
        for r in self._results:
            counts[r.predicted_action] = counts.get(r.predicted_action, 0) + 1
        return counts

    @staticmethod
    def _result_to_dict(result: ActionResult) -> dict[str, Any]:
        """Serialise an ActionResult instance to a dictionary."""
        return {
            "action_id": result.action_id,
            "sequence_id": result.sequence_id,
            "interaction_id": result.interaction_id,
            "track_id": result.track_id,
            "predicted_action": result.predicted_action,
            "action_confidence": result.action_confidence,
            "class_probabilities": result.class_probabilities,
            "top_k_predictions": [list(item) for item in result.top_k_predictions],
            "inference_time_ms": result.inference_time_ms,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "device_used": result.device_used,
            "skeleton_quality": result.skeleton_quality,
            "metadata": result.metadata,
        }

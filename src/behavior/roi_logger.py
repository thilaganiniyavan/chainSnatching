"""ROI Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``interaction_rois.json``
- ``roi_statistics.csv``
- ``roi_quality_report.md``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.interaction_roi import InteractionROI


class ROILogger:
    """Logs and exports InteractionROIs to JSON, CSV, and markdown report formats."""

    def __init__(self) -> None:
        self._rois: list[InteractionROI] = []

    def log_roi(self, roi: InteractionROI) -> None:
        """Store an InteractionROI for export."""
        self._rois.append(roi)

    def log_rois(self, rois: list[InteractionROI]) -> None:
        """Store multiple InteractionROIs for export."""
        for r in rois:
            self.log_roi(r)

    def export_json(self, output_path: str) -> None:
        """Export all logged ROIs to interaction_rois.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        roi_dicts = [self._roi_to_dict(r) for r in self._rois]

        accepted_cnt = sum(1 for r in self._rois if r.is_accepted)
        rejected_cnt = len(self._rois) - accepted_cnt

        payload = {
            "rois": roi_dicts,
            "summary": {
                "total_rois": len(self._rois),
                "accepted_rois": accepted_cnt,
                "rejected_rois": rejected_cnt,
                "acceptance_rate_pct": round((accepted_cnt / max(1, len(self._rois))) * 100.0, 2),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_csv(self, output_path: str) -> None:
        """Export all logged ROIs to roi_statistics.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "roi_id",
            "interaction_id",
            "start_frame",
            "end_frame",
            "frame_count",
            "duration_seconds",
            "person_track_id",
            "vehicle_track_id",
            "interaction_confidence",
            "is_accepted",
            "rejection_reason",
            "completeness",
            "missing_pct",
            "bounding_box_stability",
            "track_continuity",
            "frame_coverage",
            "pattern_sequence",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self._rois:
                qm = r.quality_metrics
                writer.writerow(
                    {
                        "roi_id": r.roi_id,
                        "interaction_id": r.interaction_id,
                        "start_frame": r.start_frame,
                        "end_frame": r.end_frame,
                        "frame_count": r.frame_count,
                        "duration_seconds": r.duration_seconds,
                        "person_track_id": r.person_track_id,
                        "vehicle_track_id": r.vehicle_track_id,
                        "interaction_confidence": r.interaction_confidence,
                        "is_accepted": r.is_accepted,
                        "rejection_reason": r.rejection_reason,
                        "completeness": qm.get("completeness", 0.0),
                        "missing_pct": qm.get("missing_detection_percentage", 0.0),
                        "bounding_box_stability": qm.get("bounding_box_stability", 0.0),
                        "track_continuity": qm.get("track_continuity", 0.0),
                        "frame_coverage": qm.get("frame_coverage", 0.0),
                        "pattern_sequence": " -> ".join(r.pattern_sequence),
                    }
                )

    def export_quality_report(self, output_path: str) -> None:
        """Generate formatted roi_quality_report.md."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        accepted_cnt = sum(1 for r in self._rois if r.is_accepted)
        rejected_cnt = len(self._rois) - accepted_cnt

        lines: list[str] = []
        lines.append("# Interaction ROI Quality Evaluation Report\n")
        lines.append(f"**Total ROIs Evaluated:** {len(self._rois)}")
        lines.append(f"**Accepted ROIs:** {accepted_cnt}")
        lines.append(f"**Rejected ROIs:** {rejected_cnt}")
        lines.append(f"**Acceptance Rate:** {(accepted_cnt / max(1, len(self._rois))) * 100.0:.1f}%\n")

        lines.append("## ROI Details\n")
        lines.append("| ROI ID | Interaction ID | Frames | Status | Completeness | Stability | Rejection Reason |")
        lines.append("|---|---|---|---|---|---|---|")

        for r in self._rois:
            qm = r.quality_metrics
            status = "ACCEPTED" if r.is_accepted else "REJECTED"
            comp = qm.get("completeness", 0.0)
            stab = qm.get("bounding_box_stability", 0.0)
            lines.append(
                f"| {r.roi_id} | {r.interaction_id} | {r.frame_count} ({r.duration_seconds:.1f}s) | "
                f"{status} | {comp:.0%} | {stab:.2f} | {r.rejection_reason} |"
            )

        lines.append("\n---\n*Report generated by the Interaction ROI Selection Engine.*\n")

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

    def get_rois(self) -> list[InteractionROI]:
        """Return all logged ROIs."""
        return list(self._rois)

    def clear(self) -> None:
        """Clear internal ROI log storage."""
        self._rois.clear()

    @staticmethod
    def _roi_to_dict(roi: InteractionROI) -> dict[str, Any]:
        """Serialise an InteractionROI instance to a clean dictionary."""
        return {
            "roi_id": roi.roi_id,
            "interaction_id": roi.interaction_id,
            "video_id": roi.video_id,
            "start_frame": roi.start_frame,
            "end_frame": roi.end_frame,
            "frame_count": roi.frame_count,
            "duration_seconds": roi.duration_seconds,
            "person_track_id": roi.person_track_id,
            "vehicle_track_id": roi.vehicle_track_id,
            "bounding_box_sequence": [list(b) for b in roi.bounding_box_sequence],
            "expanded_bounding_boxes": [list(b) for b in roi.expanded_bounding_boxes],
            "frame_index_mapping": roi.frame_index_mapping,
            "timestamps": roi.timestamps,
            "graph_reference_id": roi.graph_reference_id,
            "interaction_confidence": roi.interaction_confidence,
            "pattern_sequence": roi.pattern_sequence,
            "quality_metrics": roi.quality_metrics,
            "is_accepted": roi.is_accepted,
            "rejection_reason": roi.rejection_reason,
            "metadata": roi.metadata,
        }

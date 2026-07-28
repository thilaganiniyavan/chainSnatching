"""ROI Selection Research Evaluation & Statistics Framework.

Demonstrates that progressive filtering drastically reduces computational cost:
- Measures total video frames vs forwarded interaction frames to pose estimation
- Computes frame reduction percentage: ((total_frames - forwarded_frames) / total_frames) * 100
- Evaluates average ROI duration, bounding box size/area, stability, continuity
- Outputs: roi_statistics.csv, roi_selection_report.md
- Generates publication-quality plots:
  - roi_duration_histogram.png
  - roi_quality_histogram.png
  - frame_reduction_chart.png
  - interaction_length_distribution.png
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.models.interaction_roi import InteractionROI


class ROIStatisticsCollector:
    """Collects research metrics measuring progressive filtering efficiency and ROI quality."""

    def __init__(
        self,
        video_name: str = "",
        total_video_frames: int = 0,
        fps: float = 30.0,
    ) -> None:
        self.video_name = video_name
        self.total_video_frames = total_video_frames
        self.fps = fps if fps > 0 else 30.0
        self.rois: list[InteractionROI] = []

    def record_rois(self, rois: list[InteractionROI]) -> None:
        """Record InteractionROI objects."""
        self.rois.extend(rois)

    def set_total_video_frames(self, frames: int) -> None:
        """Set total video frame count for frame reduction calculation."""
        self.total_video_frames = max(0, frames)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate research evaluation metrics."""
        total_interactions = len(self.rois)
        accepted_rois = [r for r in self.rois if r.is_accepted]
        accepted_cnt = len(accepted_rois)

        # Collect unique video frame numbers forwarded to pose estimation across accepted ROIs
        forwarded_frame_indices: set[int] = set()
        for r in accepted_rois:
            forwarded_frame_indices.update(r.frame_index_mapping)

        forwarded_frames_count = len(forwarded_frame_indices)
        total_v_frames = max(self.total_video_frames, max((r.end_frame for r in self.rois), default=0))

        frame_reduction_pct = (
            ((total_v_frames - forwarded_frames_count) / max(1, total_v_frames)) * 100.0
            if total_v_frames > 0
            else 0.0
        )

        # ROI durations and box sizes
        durations_sec = [r.duration_seconds for r in accepted_rois]
        box_areas = [
            (b[2] - b[0]) * (b[3] - b[1])
            for r in accepted_rois
            for b in r.expanded_bounding_boxes
        ]

        # Quality metrics
        stabilities = [r.quality_metrics.get("bounding_box_stability", 0.0) for r in accepted_rois]
        continuities = [r.quality_metrics.get("track_continuity", 0.0) for r in accepted_rois]
        completenesses = [r.quality_metrics.get("completeness", 0.0) for r in accepted_rois]

        return {
            "video_name": self.video_name,
            "total_video_frames": total_v_frames,
            "total_interactions": total_interactions,
            "selected_accepted_rois": accepted_cnt,
            "rejected_rois": total_interactions - accepted_cnt,
            "acceptance_rate_pct": round((accepted_cnt / max(1, total_interactions)) * 100.0, 2),
            "forwarded_frames_count": forwarded_frames_count,
            "frame_reduction_pct": round(float(frame_reduction_pct), 2),
            "averages": {
                "avg_roi_duration_seconds": round(float(np.mean(durations_sec)), 3) if durations_sec else 0.0,
                "max_roi_duration_seconds": round(float(np.max(durations_sec)), 3) if durations_sec else 0.0,
                "avg_roi_box_area_px": round(float(np.mean(box_areas)), 1) if box_areas else 0.0,
                "avg_stability_score": round(float(np.mean(stabilities)), 4) if stabilities else 0.0,
                "avg_continuity_score": round(float(np.mean(continuities)), 4) if continuities else 0.0,
                "avg_completeness": round(float(np.mean(completeness)), 4) if completenesses else 0.0,
            },
        }


def save_roi_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save ROI statistics to roi_statistics.csv and roi_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "roi_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "roi_statistics.csv")
    fieldnames = [
        "video_name",
        "total_video_frames",
        "total_interactions",
        "selected_accepted_rois",
        "acceptance_rate_pct",
        "forwarded_frames_count",
        "frame_reduction_pct",
        "avg_roi_duration_seconds",
        "avg_roi_box_area_px",
        "avg_stability_score",
        "avg_continuity_score",
        "avg_completeness",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            avg = st.get("averages", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_video_frames": st.get("total_video_frames", 0),
                    "total_interactions": st.get("total_interactions", 0),
                    "selected_accepted_rois": st.get("selected_accepted_rois", 0),
                    "acceptance_rate_pct": st.get("acceptance_rate_pct", 0.0),
                    "forwarded_frames_count": st.get("forwarded_frames_count", 0),
                    "frame_reduction_pct": st.get("frame_reduction_pct", 0.0),
                    "avg_roi_duration_seconds": avg.get("avg_roi_duration_seconds", 0.0),
                    "avg_roi_box_area_px": avg.get("avg_roi_box_area_px", 0.0),
                    "avg_stability_score": avg.get("avg_stability_score", 0.0),
                    "avg_continuity_score": avg.get("avg_continuity_score", 0.0),
                    "avg_completeness": avg.get("avg_completeness", 0.0),
                }
            )


def generate_publication_figures(
    all_rois: List[InteractionROI],
    all_stats: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Generate publication-quality research plots:

    - frame_reduction_chart.png
    - roi_duration_histogram.png
    - roi_quality_histogram.png
    - interaction_length_distribution.png
    """
    os.makedirs(output_dir, exist_ok=True)
    accepted_rois = [r for r in all_rois if r.is_accepted]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Frame Reduction Comparison Bar Chart
    if all_stats:
        video_names = [st.get("video_name", f"Vid_{i}") for i, st in enumerate(all_stats)]
        tot_frames = [st.get("total_video_frames", 0) for st in all_stats]
        fwd_frames = [st.get("forwarded_frames_count", 0) for st in all_stats]

        x = np.arange(len(video_names))
        width = 0.35

        plt.figure(figsize=(9, 5))
        plt.bar(x - width/2, tot_frames, width, label="Baseline (Full Video Frames)", color="#d95f02")
        plt.bar(x + width/2, fwd_frames, width, label="Proposed ROI Selection Frames", color="#1b9e77")

        plt.title("Computational Workload Reduction: Baseline vs Proposed ROI Selection", fontsize=13, fontweight="bold")
        plt.xlabel("Video Sequence", fontsize=11)
        plt.ylabel("Processed Frames", fontsize=11)
        plt.xticks(x, video_names, rotation=15)
        plt.legend(fontsize=10)

        for i in range(len(video_names)):
            red_pct = all_stats[i].get("frame_reduction_pct", 0.0)
            plt.text(x[i], max(tot_frames[i], fwd_frames[i]) * 1.02, f"-{red_pct:.1f}%", ha="center", fontweight="bold", color="#1b9e77")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "frame_reduction_chart.png"), dpi=300)
        plt.close()

    # 2. ROI Duration Histogram
    if accepted_rois:
        durations = [r.duration_seconds for r in accepted_rois]
        plt.figure(figsize=(8, 5))
        plt.hist(durations, bins=10, color="#2b5c8f", edgecolor="black", alpha=0.85)
        plt.title("Selected ROI Duration Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Duration (seconds)", fontsize=12)
        plt.ylabel("Accepted ROI Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roi_duration_histogram.png"), dpi=300)
        plt.close()

    # 3. ROI Quality Scores Histogram
    if accepted_rois:
        stabilities = [r.quality_metrics.get("bounding_box_stability", 0.0) for r in accepted_rois]
        completenesses = [r.quality_metrics.get("completeness", 0.0) for r in accepted_rois]

        plt.figure(figsize=(9, 5))
        plt.hist(stabilities, bins=10, range=(0.0, 1.0), alpha=0.7, label="BBox Stability", color="#02818a", edgecolor="black")
        plt.hist(completenesses, bins=10, range=(0.0, 1.0), alpha=0.7, label="Detection Completeness", color="#67a9cf", edgecolor="black")

        plt.title("ROI Quality Metric Distributions", fontsize=14, fontweight="bold")
        plt.xlabel("Score [0.0 - 1.0]", fontsize=12)
        plt.ylabel("ROI Count", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roi_quality_histogram.png"), dpi=300)
        plt.close()

    # 4. Interaction Length Distribution (Frames)
    if all_rois:
        frame_lengths = [r.frame_count for r in all_rois]
        plt.figure(figsize=(8, 5))
        plt.hist(frame_lengths, bins=12, color="#7570b3", edgecolor="black", alpha=0.85)
        plt.title("Interaction Frame Length Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Interaction Frame Span", fontsize=12)
        plt.ylabel("Interaction Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "interaction_length_distribution.png"), dpi=300)
        plt.close()


def generate_roi_selection_report(
    all_stats: List[Dict[str, Any]],
    all_rois: List[InteractionROI],
    output_dir: str,
) -> None:
    """Generate roi_selection_report.md containing research evaluation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "roi_selection_report.md")

    lines: list[str] = []
    lines.append("# Interaction ROI Selection Engine — Research Evaluation Report\n")
    lines.append(f"**Total Videos Evaluated:** {len(all_stats)}")
    lines.append(f"**Total Candidate Interactions:** {len(all_rois)}")

    accepted_cnt = sum(1 for r in all_rois if r.is_accepted)
    lines.append(f"**Accepted ROIs:** {accepted_cnt}")
    lines.append(f"**Rejected ROIs:** {len(all_rois) - accepted_cnt}\n")

    # Section 1: Computational Reduction Summary
    lines.append("## Computational Workload Reduction Summary\n")
    lines.append("| Video Name | Total Video Frames | Forwarded ROI Frames | Frame Reduction (%) | Acceptance Rate (%) |")
    lines.append("|---|---|---|---|---|")

    for st in all_stats:
        lines.append(
            f"| {st.get('video_name', '—')} | {st.get('total_video_frames', 0)} | "
            f"{st.get('forwarded_frames_count', 0)} | **{st.get('frame_reduction_pct', 0.0):.1f}%** | "
            f"{st.get('acceptance_rate_pct', 0.0):.1f}% |"
        )

    # Section 2: Quality & Metric Breakdown
    lines.append("\n## Selected ROI Quality Metrics\n")
    lines.append("| ROI ID | Duration (s) | Completeness | Stability | Continuity | Status |")
    lines.append("|---|---|---|---|---|---|")

    for r in all_rois:
        qm = r.quality_metrics
        status = "ACCEPTED" if r.is_accepted else "REJECTED"
        comp = qm.get("completeness", 0.0)
        stab = qm.get("bounding_box_stability", 0.0)
        cont = qm.get("track_continuity", 0.0)
        lines.append(
            f"| {r.roi_id} | {r.duration_seconds:.1f}s | {comp:.0%} | {stab:.2f} | {cont:.2f} | {status} |"
        )

    lines.append("\n---\n*Report generated by the Interaction ROI Research Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

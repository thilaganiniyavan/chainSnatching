"""Skeleton Sequence Research Evaluation & Statistics Framework.

Measures:
- Average sequence length (frames & seconds)
- Sequence completeness & continuity
- Missing joint percentage
- Average keypoint confidence
- Joint stability score
- Sequence rejection rate
- Average tensor generation time (ms)

Outputs:
- sequence_statistics.csv
- sequence_quality_report.md
- Publication-quality figures:
  - sequence_length_histogram.png
  - joint_confidence_distribution.png
  - missing_joint_heatmap.png
  - sequence_quality_histogram.png
  - temporal_completeness_chart.png
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

from src.core.models.skeleton_sequence import SkeletonSequence


class SequenceStatisticsCollector:
    """Collects research metrics measuring skeleton sequence quality and tensor construction performance."""

    def __init__(self, video_name: str = "", fps: float = 30.0) -> None:
        self.video_name = video_name
        self.fps = fps if fps > 0 else 30.0
        self.sequences: list[SkeletonSequence] = []

    def record_sequences(self, sequences: list[SkeletonSequence]) -> None:
        """Record SkeletonSequence objects."""
        self.sequences.extend(sequences)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate research evaluation metrics."""
        total_seqs = len(self.sequences)
        accepted_seqs = [s for s in self.sequences if s.is_accepted]
        accepted_cnt = len(accepted_seqs)

        if total_seqs == 0:
            return {
                "video_name": self.video_name,
                "total_sequences": 0,
                "accepted_sequences": 0,
                "rejection_rate_pct": 0.0,
            }

        rejection_rate = ((total_seqs - accepted_cnt) / total_seqs) * 100.0

        lengths_frames = [s.frame_count for s in accepted_seqs]
        durations_sec = [s.duration_seconds for s in accepted_seqs]
        quality_scores = [s.quality_score for s in accepted_seqs]
        completeness_scores = [s.completeness_score for s in accepted_seqs]

        # Calculate missing joint percentage (< 0.3 conf)
        total_joints = sum(s.joint_confidence_matrix.size for s in accepted_seqs)
        missing_joints = sum(
            int(np.sum(s.joint_confidence_matrix < 0.3)) for s in accepted_seqs
        )
        missing_joint_pct = (missing_joints / max(1, total_joints)) * 100.0

        confidences = [
            float(np.mean(s.joint_confidence_matrix))
            for s in accepted_seqs
            if s.joint_confidence_matrix.size > 0
        ]

        return {
            "video_name": self.video_name,
            "total_sequences": total_seqs,
            "accepted_sequences": accepted_cnt,
            "rejected_sequences": total_seqs - accepted_cnt,
            "rejection_rate_pct": round(rejection_rate, 2),
            "performance": {
                "avg_sequence_length_frames": round(float(np.mean(lengths_frames)), 1) if lengths_frames else 0.0,
                "avg_sequence_duration_sec": round(float(np.mean(durations_sec)), 2) if durations_sec else 0.0,
                "avg_confidence_score": round(float(np.mean(confidences)), 4) if confidences else 0.0,
                "missing_joint_pct": round(float(missing_joint_pct), 2),
                "avg_quality_score": round(float(np.mean(quality_scores)), 4) if quality_scores else 0.0,
                "avg_completeness": round(float(np.mean(completeness_scores)), 4) if completeness_scores else 0.0,
            },
        }


def save_sequence_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save sequence statistics to sequence_statistics.csv and sequence_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "sequence_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "sequence_statistics.csv")
    fieldnames = [
        "video_name",
        "total_sequences",
        "accepted_sequences",
        "rejected_sequences",
        "rejection_rate_pct",
        "avg_sequence_length_frames",
        "avg_sequence_duration_sec",
        "avg_confidence_score",
        "missing_joint_pct",
        "avg_quality_score",
        "avg_completeness",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            perf = st.get("performance", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_sequences": st.get("total_sequences", 0),
                    "accepted_sequences": st.get("accepted_sequences", 0),
                    "rejected_sequences": st.get("rejected_sequences", 0),
                    "rejection_rate_pct": st.get("rejection_rate_pct", 0.0),
                    "avg_sequence_length_frames": perf.get("avg_sequence_length_frames", 0.0),
                    "avg_sequence_duration_sec": perf.get("avg_sequence_duration_sec", 0.0),
                    "avg_confidence_score": perf.get("avg_confidence_score", 0.0),
                    "missing_joint_pct": perf.get("missing_joint_pct", 0.0),
                    "avg_quality_score": perf.get("avg_quality_score", 0.0),
                    "avg_completeness": perf.get("avg_completeness", 0.0),
                }
            )


def generate_publication_figures(
    all_seqs: List[SkeletonSequence],
    all_stats: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Generate publication-quality research figures:

    - sequence_length_histogram.png
    - joint_confidence_distribution.png
    - missing_joint_heatmap.png
    - sequence_quality_histogram.png
    - temporal_completeness_chart.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_seqs:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    accepted_seqs = [s for s in all_seqs if s.is_accepted]

    # 1. Sequence Length Histogram
    if accepted_seqs:
        lengths = [s.frame_count for s in accepted_seqs]
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=10, color="#1b9e77", edgecolor="black", alpha=0.85)
        plt.title("Skeleton Sequence Frame Length Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Sequence Length (Frames)", fontsize=12)
        plt.ylabel("Accepted Sequence Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sequence_length_histogram.png"), dpi=300)
        plt.close()

    # 2. Joint Confidence Distribution
    if accepted_seqs:
        all_confs = [
            float(np.mean(s.joint_confidence_matrix))
            for s in accepted_seqs
            if s.joint_confidence_matrix.size > 0
        ]
        plt.figure(figsize=(8, 5))
        plt.hist(all_confs, bins=10, range=(0.0, 1.0), color="#7570b3", edgecolor="black", alpha=0.85)
        plt.title("Average Sequence Keypoint Confidence Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Mean Keypoint Confidence", fontsize=12)
        plt.ylabel("Sequence Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "joint_confidence_distribution.png"), dpi=300)
        plt.close()

    # 3. Missing Joint Heatmap across Joint Indices (T x V)
    if accepted_seqs and accepted_seqs[0].joint_confidence_matrix.size > 0:
        sample_conf = accepted_seqs[0].joint_confidence_matrix.T # (V, T)
        plt.figure(figsize=(9, 5))
        plt.imshow(sample_conf, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Joint Confidence")
        plt.title(f"Joint Confidence Heatmap Across Time ({accepted_seqs[0].sequence_id})", fontsize=13, fontweight="bold")
        plt.xlabel("Frame Index T", fontsize=11)
        plt.ylabel("Joint Index V", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_joint_heatmap.png"), dpi=300)
        plt.close()

    # 4. Sequence Quality Score Histogram
    if all_seqs:
        qualities = [s.quality_score for s in all_seqs]
        plt.figure(figsize=(8, 5))
        plt.hist(qualities, bins=10, range=(0.0, 1.0), color="#d95f02", edgecolor="black", alpha=0.85)
        plt.title("Overall Sequence Quality Score Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Quality Score [0.0 - 1.0]", fontsize=12)
        plt.ylabel("Sequence Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sequence_quality_histogram.png"), dpi=300)
        plt.close()

    # 5. Temporal Completeness Chart
    if accepted_seqs:
        completeness_scores = [s.completeness_score for s in accepted_seqs]
        plt.figure(figsize=(8, 5))
        plt.hist(completeness_scores, bins=10, range=(0.0, 1.0), color="#2b5c8f", edgecolor="black", alpha=0.85)
        plt.title("Sequence Temporal Completeness Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Completeness Ratio", fontsize=12)
        plt.ylabel("Sequence Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temporal_completeness_chart.png"), dpi=300)
        plt.close()


def generate_sequence_quality_report(
    all_stats: List[Dict[str, Any]],
    all_seqs: List[SkeletonSequence],
    output_dir: str,
) -> None:
    """Generate sequence_quality_report.md containing research evaluation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "sequence_quality_report.md")

    lines: list[str] = []
    lines.append("# Skeleton Sequence Builder — Research Evaluation Report\n")
    lines.append(f"**Total Videos Evaluated:** {len(all_stats)}")
    lines.append(f"**Total Sequences Generated:** {len(all_seqs)}")

    accepted_cnt = sum(1 for s in all_seqs if s.is_accepted)
    lines.append(f"**Accepted Sequences:** {accepted_cnt}")
    lines.append(f"**Rejected Sequences:** {len(all_seqs) - accepted_cnt}\n")

    # Section 1: Performance Summary Table
    lines.append("## Sequence Construction Summary\n")
    lines.append("| Video Name | Total Seqs | Accepted | Rejection Rate (%) | Avg Length (f) | Avg Confidence | Quality Score |")
    lines.append("|---|---|---|---|---|---|---|")

    for st in all_stats:
        perf = st.get("performance", {})
        lines.append(
            f"| {st.get('video_name', '—')} | {st.get('total_sequences', 0)} | "
            f"{st.get('accepted_sequences', 0)} | **{st.get('rejection_rate_pct', 0.0):.1f}%** | "
            f"{perf.get('avg_sequence_length_frames', 0.0):.1f}f | {perf.get('avg_confidence_score', 0.0):.0%} | "
            f"{perf.get('avg_quality_score', 0.0):.2f} |"
        )

    # Section 2: Sequence Entry Breakdown
    lines.append("\n## Sequence Details\n")
    lines.append("| Sequence ID | Interaction ID | Track ID | Frames | Normalization | Quality | Status |")
    lines.append("|---|---|---|---|---|---|---|")

    for s in all_seqs[:20]:
        status = "ACCEPTED" if s.is_accepted else "REJECTED"
        lines.append(
            f"| {s.sequence_id} | {s.interaction_id} | {s.person_track_id} | "
            f"{s.frame_count} ({s.duration_seconds:.1f}s) | {s.normalization_method} | {s.quality_score:.2f} | {status} |"
        )

    lines.append("\n---\n*Report generated by the Skeleton Sequence Research Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

"""Pose Estimation Research Evaluation & Statistics Framework.

Measures:
- Pose inference runtime (ms/frame, ms/ROI)
- Inference throughput (FPS)
- Total pose samples processed
- Average keypoint confidence
- Missing keypoint percentage
- Keypoint stability (jitter variance)
- Skeleton quality score distribution
- GPU utilization (PyTorch CUDA status check or CPU fallback)

Outputs:
- pose_statistics.csv
- pose_quality_report.md
- Publication-quality figures:
  - pose_confidence_histogram.png
  - pose_runtime_benchmark.png
  - skeleton_quality_distribution.png
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

from src.core.models.pose_result import PoseResult

# Try checking PyTorch CUDA GPU availability
HAS_TORCH_CUDA = False
GPU_NAME = "N/A (CPU Mode)"
try:
    import torch
    if torch.cuda.is_available():
        HAS_TORCH_CUDA = True
        GPU_NAME = torch.cuda.get_device_name(0)
except Exception:
    HAS_TORCH_CUDA = False


class PoseStatisticsCollector:
    """Collects research evaluation metrics for pose estimation across executions."""

    def __init__(self, video_name: str = "", fps: float = 30.0) -> None:
        self.video_name = video_name
        self.fps = fps if fps > 0 else 30.0
        self.poses: list[PoseResult] = []

    def record_poses(self, poses: list[PoseResult]) -> None:
        """Record PoseResult instances."""
        self.poses.extend(poses)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate research metrics across all recorded pose results."""
        total_samples = len(self.poses)
        if total_samples == 0:
            return {
                "video_name": self.video_name,
                "total_pose_samples": 0,
                "gpu_available": HAS_TORCH_CUDA,
                "gpu_device": GPU_NAME,
            }

        confidences = [p.overall_confidence for p in self.poses]
        quality_scores = [p.quality_score for p in self.poses]
        runtimes = [p.processing_time_ms for p in self.poses]

        # Compute missing keypoint % (< 0.3 conf)
        total_kp = sum(p.num_keypoints for p in self.poses)
        missing_kp = sum(
            1 for p in self.poses for kp in p.keypoints_pixel if kp[2] < 0.3
        )
        missing_pct = (missing_kp / max(1, total_kp)) * 100.0

        # Throughput FPS (1000.0 / avg_ms)
        avg_runtime_ms = float(np.mean(runtimes)) if runtimes else 0.0
        throughput_fps = (1000.0 / avg_runtime_ms) if avg_runtime_ms > 0 else 0.0

        # Keypoint jitter stability (std-dev of per-joint displacements)
        jitter_stds: list[float] = []
        for i in range(1, len(self.poses)):
            p1 = self.poses[i - 1]
            p2 = self.poses[i]
            if p1.track_id == p2.track_id and p1.num_keypoints == p2.num_keypoints:
                diffs = []
                for k in range(p1.num_keypoints):
                    kp1 = np.array(p1.keypoints_pixel[k][:2])
                    kp2 = np.array(p2.keypoints_pixel[k][:2])
                    diffs.append(np.linalg.norm(kp2 - kp1))
                jitter_stds.append(float(np.mean(diffs)))

        avg_jitter = float(np.mean(jitter_stds)) if jitter_stds else 0.0

        backend_counts: Dict[str, int] = defaultdict(int)
        for p in self.poses:
            backend_counts[p.backend_name] += 1

        return {
            "video_name": self.video_name,
            "total_pose_samples": total_samples,
            "gpu_available": HAS_TORCH_CUDA,
            "gpu_device": GPU_NAME,
            "backend_counts": dict(backend_counts),
            "performance": {
                "avg_inference_time_ms": round(avg_runtime_ms, 2),
                "throughput_fps": round(float(throughput_fps), 1),
                "avg_keypoint_confidence": round(float(np.mean(confidences)), 4),
                "missing_keypoint_pct": round(float(missing_pct), 2),
                "avg_quality_score": round(float(np.mean(quality_scores)), 4),
                "keypoint_jitter_px": round(avg_jitter, 2),
            },
        }


def save_pose_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save pose statistics to pose_statistics.csv and pose_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "pose_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "pose_statistics.csv")
    fieldnames = [
        "video_name",
        "total_pose_samples",
        "gpu_available",
        "gpu_device",
        "avg_inference_time_ms",
        "throughput_fps",
        "avg_keypoint_confidence",
        "missing_keypoint_pct",
        "avg_quality_score",
        "keypoint_jitter_px",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            perf = st.get("performance", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_pose_samples": st.get("total_pose_samples", 0),
                    "gpu_available": st.get("gpu_available", False),
                    "gpu_device": st.get("gpu_device", "N/A"),
                    "avg_inference_time_ms": perf.get("avg_inference_time_ms", 0.0),
                    "throughput_fps": perf.get("throughput_fps", 0.0),
                    "avg_keypoint_confidence": perf.get("avg_keypoint_confidence", 0.0),
                    "missing_keypoint_pct": perf.get("missing_keypoint_pct", 0.0),
                    "avg_quality_score": perf.get("avg_quality_score", 0.0),
                    "keypoint_jitter_px": perf.get("keypoint_jitter_px", 0.0),
                }
            )


def generate_publication_figures(
    all_poses: List[PoseResult],
    all_stats: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Generate publication-quality research figures:

    - pose_confidence_histogram.png
    - pose_runtime_benchmark.png
    - skeleton_quality_distribution.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_poses:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Pose Keypoint Confidence Histogram
    confidences = [p.overall_confidence for p in all_poses]
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=10, range=(0.0, 1.0), color="#1b9e77", edgecolor="black", alpha=0.85)
    plt.title("Pose Overall Keypoint Confidence Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Keypoint Confidence Score", fontsize=12)
    plt.ylabel("Pose Samples Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pose_confidence_histogram.png"), dpi=300)
    plt.close()

    # 2. Pose Inference Runtime Benchmark Bar Chart per Backend
    backend_times: Dict[str, list[float]] = defaultdict(list)
    for p in all_poses:
        backend_times[p.backend_name].append(p.processing_time_ms)

    backends = sorted(backend_times.keys())
    avg_runtimes = [float(np.mean(backend_times[b])) for b in backends]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(backends, avg_runtimes, color="#2b5c8f", width=0.45)
    plt.title("Pose Backend Inference Runtime Benchmark (ms/frame)", fontsize=14, fontweight="bold")
    plt.xlabel("Pose Estimation Backend Adapter", fontsize=12)
    plt.ylabel("Inference Time (ms)", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pose_runtime_benchmark.png"), dpi=300)
    plt.close()

    # 3. Skeleton Quality Score Distribution
    quality_scores = [p.quality_score for p in all_poses]
    plt.figure(figsize=(8, 5))
    plt.hist(quality_scores, bins=10, range=(0.0, 1.0), color="#d95f02", edgecolor="black", alpha=0.85)
    plt.title("Skeleton Quality Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Skeleton Quality Score (Confidence × Completeness)", fontsize=12)
    plt.ylabel("Pose Samples Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skeleton_quality_distribution.png"), dpi=300)
    plt.close()


def generate_pose_quality_report(
    all_stats: List[Dict[str, Any]],
    all_poses: List[PoseResult],
    output_dir: str,
) -> None:
    """Generate pose_quality_report.md containing research evaluation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pose_quality_report.md")

    lines: list[str] = []
    lines.append("# Pose Estimation Abstraction Layer — Research Evaluation Report\n")
    lines.append(f"**Total Videos Evaluated:** {len(all_stats)}")
    lines.append(f"**Total Pose Samples Extracted:** {len(all_poses)}")

    if all_stats:
        gpu_str = all_stats[0].get("gpu_device", "N/A")
        lines.append(f"**Hardware Environment:** {gpu_str}\n")

    # Section 1: Performance Summary Table
    lines.append("## Pose Backend Performance Summary\n")
    lines.append("| Video Name | Samples | Avg Runtime (ms) | Throughput (FPS) | Avg Confidence | Missing Keypoint % | Quality Score |")
    lines.append("|---|---|---|---|---|---|---|")

    for st in all_stats:
        perf = st.get("performance", {})
        lines.append(
            f"| {st.get('video_name', '—')} | {st.get('total_pose_samples', 0)} | "
            f"{perf.get('avg_inference_time_ms', 0.0):.2f}ms | {perf.get('throughput_fps', 0.0):.1f} FPS | "
            f"{perf.get('avg_keypoint_confidence', 0.0):.0%} | {perf.get('missing_keypoint_pct', 0.0):.1f}% | "
            f"{perf.get('avg_quality_score', 0.0):.2f} |"
        )

    # Section 2: Sample Pose Result Entries
    lines.append("\n## Sample Extracted Poses\n")
    lines.append("| Sample ID | Frame | Track ID | Backend | Topology | Confidence | Quality Score |")
    lines.append("|---|---|---|---|---|---|---|")

    for p in all_poses[:15]:
        lines.append(
            f"| {p.sample_id} | {p.frame_index} | {p.track_id} | "
            f"{p.backend_name} | {p.topology} | {p.overall_confidence:.0%} | {p.quality_score:.2f} |"
        )

    lines.append("\n---\n*Report generated by the Pose Estimation Research Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

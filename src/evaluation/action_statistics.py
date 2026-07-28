"""Human Action Recognition Research Evaluation & Statistics Framework.

Measures:
- Action classification latency (ms/sequence & FPS)
- Class prediction frequency distribution
- Action confidence distribution
- GPU vs CPU hardware execution performance
- Model throughput

Outputs:
- action_statistics.csv
- action_recognition_report.md
- Publication-quality figures:
  - action_frequency_chart.png
  - action_confidence_histogram.png
  - action_latency_histogram.png
  - runtime_comparison_chart.png
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

from src.core.models.action_result import ActionResult


class ActionStatisticsCollector:
    """Collects research evaluation metrics for human action classification."""

    def __init__(self, video_name: str = "") -> None:
        self.video_name = video_name
        self.results: list[ActionResult] = []

    def record_results(self, results: list[ActionResult]) -> None:
        """Record ActionResult objects."""
        self.results.extend(results)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate research evaluation metrics."""
        total_evals = len(self.results)
        if total_evals == 0:
            return {
                "video_name": self.video_name,
                "total_action_classifications": 0,
            }

        confidences = [r.action_confidence for r in self.results]
        runtimes = [r.inference_time_ms for r in self.results]

        avg_latency_ms = float(np.mean(runtimes)) if runtimes else 0.0
        throughput_fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

        action_counts: Dict[str, int] = defaultdict(int)
        for r in self.results:
            action_counts[r.predicted_action] += 1

        device_counts: Dict[str, int] = defaultdict(int)
        for r in self.results:
            device_counts[r.device_used] += 1

        return {
            "video_name": self.video_name,
            "total_action_classifications": total_evals,
            "device_counts": dict(device_counts),
            "action_counts": dict(action_counts),
            "performance": {
                "avg_inference_latency_ms": round(avg_latency_ms, 2),
                "throughput_fps": round(float(throughput_fps), 1),
                "avg_action_confidence": round(float(np.mean(confidences)), 4),
            },
        }


def save_action_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save action statistics to action_statistics.csv and action_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "action_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "action_statistics.csv")
    fieldnames = [
        "video_name",
        "total_action_classifications",
        "avg_inference_latency_ms",
        "throughput_fps",
        "avg_action_confidence",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            perf = st.get("performance", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_action_classifications": st.get("total_action_classifications", 0),
                    "avg_inference_latency_ms": perf.get("avg_inference_latency_ms", 0.0),
                    "throughput_fps": perf.get("throughput_fps", 0.0),
                    "avg_action_confidence": perf.get("avg_action_confidence", 0.0),
                }
            )


def generate_publication_figures(
    all_results: List[ActionResult],
    all_stats: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Generate publication-quality research figures:

    - action_frequency_chart.png
    - action_confidence_histogram.png
    - action_latency_histogram.png
    - runtime_comparison_chart.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_results:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Action Frequency Bar Chart
    action_counts: Dict[str, int] = defaultdict(int)
    for r in all_results:
        action_counts[r.predicted_action] += 1

    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(actions, counts, color="#2b5c8f", width=0.5, edgecolor="black")
    plt.title("Action Recognition Class Prediction Frequency", fontsize=14, fontweight="bold")
    plt.xlabel("Action Class", fontsize=12)
    plt.ylabel("Classification Count", fontsize=12)
    plt.xticks(rotation=20)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_frequency_chart.png"), dpi=300)
    plt.close()

    # 2. Action Prediction Confidence Histogram
    confidences = [r.action_confidence for r in all_results]
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=10, range=(0.0, 1.0), color="#1b9e77", edgecolor="black", alpha=0.85)
    plt.title("Action Classification Confidence Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Action Confidence Score", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_confidence_histogram.png"), dpi=300)
    plt.close()

    # 3. Action Inference Latency Histogram
    runtimes = [r.inference_time_ms for r in all_results]
    plt.figure(figsize=(8, 5))
    plt.hist(runtimes, bins=10, color="#d95f02", edgecolor="black", alpha=0.85)
    plt.title("Action Recognition Model Latency Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Inference Time (ms)", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_latency_histogram.png"), dpi=300)
    plt.close()

    # 4. Runtime Benchmark Comparison per Model Backend
    model_times: Dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        model_times[r.model_name].append(r.inference_time_ms)

    models = sorted(model_times.keys())
    avg_times = [float(np.mean(model_times[m])) for m in models]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, avg_times, color="#7570b3", width=0.4, edgecolor="black")
    plt.title("Action Recognition Backend Runtime Comparison (ms)", fontsize=14, fontweight="bold")
    plt.xlabel("Model Architecture Backend", fontsize=12)
    plt.ylabel("Average Latency (ms)", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison_chart.png"), dpi=300)
    plt.close()


def generate_action_recognition_report(
    all_stats: List[Dict[str, Any]],
    all_results: List[ActionResult],
    output_dir: str,
) -> None:
    """Generate action_recognition_report.md containing research evaluation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "action_recognition_report.md")

    lines: list[str] = []
    lines.append("# Human Action Recognition — Research Evaluation Report\n")
    lines.append(f"**Total Videos Evaluated:** {len(all_stats)}")
    lines.append(f"**Total Action Predictions:** {len(all_results)}")

    avg_conf = sum(r.action_confidence for r in all_results) / max(1, len(all_results))
    avg_lat = sum(r.inference_time_ms for r in all_results) / max(1, len(all_results))
    lines.append(f"**Average Prediction Confidence:** {avg_conf:.0%}")
    lines.append(f"**Average Model Latency:** {avg_lat:.2f} ms/sequence\n")

    # Section 1: Summary Table
    lines.append("## Action Recognition Summary\n")
    lines.append("| Video Name | Total Classifications | Avg Latency (ms) | Throughput (FPS) | Avg Confidence |")
    lines.append("|---|---|---|---|---|")

    for st in all_stats:
        perf = st.get("performance", {})
        lines.append(
            f"| {st.get('video_name', '—')} | {st.get('total_action_classifications', 0)} | "
            f"{perf.get('avg_inference_latency_ms', 0.0):.2f}ms | {perf.get('throughput_fps', 0.0):.1f} FPS | "
            f"{perf.get('avg_action_confidence', 0.0):.0%} |"
        )

    # Section 2: Detailed Prediction Rows
    lines.append("\n## Detailed Action Predictions\n")
    lines.append("| Action ID | Sequence ID | Track ID | Predicted Action | Confidence | Model | Latency (ms) |")
    lines.append("|---|---|---|---|---|---|---|")

    for r in all_results[:20]:
        lines.append(
            f"| {r.action_id} | {r.sequence_id} | {r.track_id} | "
            f"**{r.predicted_action}** | {r.action_confidence:.0%} | {r.model_name} | {r.inference_time_ms:.1f}ms |"
        )

    lines.append("\n---\n*Report generated by the Action Recognition Research Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

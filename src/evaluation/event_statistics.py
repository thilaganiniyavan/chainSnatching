"""Event Statistics & Research Evaluation Module.

Measures:
- Most common event types (frequencies)
- Average event duration
- Behaviour primitive to event transition frequencies
- Event confidence distribution
- Generates publication-quality plots (matplotlib)
- Generates event_statistics.csv and event_reasoning_report.md
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

from src.core.models.behaviour_event import BehaviourEvent


class EventStatisticsCollector:
    """Collects research evaluation metrics for Behaviour Events across single or multiple videos."""

    def __init__(self, video_name: str = "", fps: float = 30.0) -> None:
        self.video_name = video_name
        self.fps = fps if fps > 0 else 30.0
        self.events: list[BehaviourEvent] = []

    def record_events(self, events: list[BehaviourEvent]) -> None:
        """Record events from processing."""
        self.events.extend(events)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate all collected event metrics into a summary dictionary."""

        type_counts: Dict[str, int] = defaultdict(int)
        type_durations: Dict[str, list[float]] = defaultdict(list)
        type_confidences: Dict[str, list[float]] = defaultdict(list)
        primitive_transitions: Dict[tuple[str, str], int] = defaultdict(int)

        for event in self.events:
            etype = event.event_type
            type_counts[etype] += 1
            type_durations[etype].append(event.duration_seconds)
            type_confidences[etype].append(event.confidence)

            # Primitive -> Event transition analysis
            if event.supporting_sequence:
                for prim in event.supporting_sequence:
                    primitive_transitions[(prim, etype)] += 1

        summary_by_type: Dict[str, Any] = {}
        for etype, count in type_counts.items():
            durs = type_durations[etype]
            confs = type_confidences[etype]
            summary_by_type[etype] = {
                "count": count,
                "avg_duration_seconds": round(float(np.mean(durs)), 3) if durs else 0.0,
                "min_duration_seconds": round(float(np.min(durs)), 3) if durs else 0.0,
                "max_duration_seconds": round(float(np.max(durs)), 3) if durs else 0.0,
                "avg_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            }

        trans_matrix: Dict[str, Dict[str, int]] = {}
        for (prim, etype), cnt in primitive_transitions.items():
            if prim not in trans_matrix:
                trans_matrix[prim] = {}
            trans_matrix[prim][etype] = cnt

        all_durations = [e.duration_seconds for e in self.events]
        all_confidences = [e.confidence for e in self.events]

        return {
            "video_name": self.video_name,
            "total_events": len(self.events),
            "summary_by_type": summary_by_type,
            "overall": {
                "avg_duration_seconds": round(float(np.mean(all_durations)), 3) if all_durations else 0.0,
                "avg_confidence": round(float(np.mean(all_confidences)), 4) if all_confidences else 0.0,
            },
            "primitive_to_event_transitions": trans_matrix,
        }


def save_event_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save event statistics to event_statistics.csv and event_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "event_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "event_statistics.csv")
    fieldnames = [
        "video_name",
        "total_events",
        "avg_duration_seconds",
        "avg_confidence",
        "normal_passing_count",
        "vehicle_waiting_count",
        "following_behaviour_count",
        "stationary_interaction_count",
        "close_encounter_count",
        "suspicious_encounter_count",
        "rapid_escape_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            by_type = st.get("summary_by_type", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_events": st.get("total_events", 0),
                    "avg_duration_seconds": st.get("overall", {}).get("avg_duration_seconds", 0.0),
                    "avg_confidence": st.get("overall", {}).get("avg_confidence", 0.0),
                    "normal_passing_count": by_type.get("NORMAL_PASSING", {}).get("count", 0),
                    "vehicle_waiting_count": by_type.get("VEHICLE_WAITING", {}).get("count", 0),
                    "following_behaviour_count": by_type.get("FOLLOWING_BEHAVIOUR", {}).get("count", 0),
                    "stationary_interaction_count": by_type.get("STATIONARY_INTERACTION", {}).get("count", 0),
                    "close_encounter_count": by_type.get("CLOSE_ENCOUNTER", {}).get("count", 0),
                    "suspicious_encounter_count": by_type.get("SUSPICIOUS_ENCOUNTER", {}).get("count", 0),
                    "rapid_escape_count": by_type.get("RAPID_ESCAPE", {}).get("count", 0),
                }
            )


def generate_publication_plots(
    all_events: List[BehaviourEvent], output_dir: str
) -> None:
    """Generate publication-quality research plots:

    - event_frequency.png
    - event_duration_distribution.png
    - event_confidence_distribution.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_events:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Event Frequencies
    type_counts: Dict[str, int] = defaultdict(int)
    for e in all_events:
        type_counts[e.event_type] += 1

    types = sorted(type_counts.keys())
    counts = [type_counts[t] for t in types]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(types, counts, color="#2b5c8f", width=0.55)
    plt.title("Behaviour Event Type Frequency", fontsize=14, fontweight="bold")
    plt.xlabel("Event Type", fontsize=12)
    plt.ylabel("Occurrences", fontsize=12)
    plt.xticks(rotation=20, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, str(int(yval)), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "event_frequency.png"), dpi=300)
    plt.close()

    # 2. Confidence Distribution Histogram
    confidences = [e.confidence for e in all_events]
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=10, range=(0.0, 1.0), color="#02818a", edgecolor="black", alpha=0.8)
    plt.title("Event Classification Confidence Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "event_confidence_distribution.png"), dpi=300)
    plt.close()

    # 3. Duration Boxplot per Event Type
    durations_by_type: Dict[str, list[float]] = defaultdict(list)
    for e in all_events:
        durations_by_type[e.event_type].append(e.duration_seconds)

    if durations_by_type:
        plot_types = sorted(durations_by_type.keys())
        data = [durations_by_type[t] for t in plot_types]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, tick_labels=plot_types, patch_artist=True, boxprops=dict(facecolor="#67a9cf"))
        plt.title("Interaction Duration per Event Type (Seconds)", fontsize=14, fontweight="bold")
        plt.ylabel("Duration (s)", fontsize=12)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "event_duration_distribution.png"), dpi=300)
        plt.close()


def generate_event_reasoning_report(
    all_stats: List[Dict[str, Any]],
    all_events: List[BehaviourEvent],
    output_dir: str,
) -> None:
    """Generate event_reasoning_report.md containing research analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "event_reasoning_report.md")

    lines: list[str] = []
    lines.append("# Behaviour Reasoning Engine — Research Evaluation Report\n")
    lines.append(f"**Total Videos Analysed:** {len(all_stats)}")
    lines.append(f"**Total Behaviour Events Detected:** {len(all_events)}\n")

    # Table 1: Summary by Event Type
    type_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "durations": [], "confidences": []})
    for e in all_events:
        type_stats[e.event_type]["count"] += 1
        type_stats[e.event_type]["durations"].append(e.duration_seconds)
        type_stats[e.event_type]["confidences"].append(e.confidence)

    lines.append("## Event Summary & Metrics\n")
    lines.append("| Event Type | Count | Avg Duration (s) | Max Duration (s) | Avg Confidence |")
    lines.append("|---|---|---|---|---|")

    for etype in sorted(type_stats.keys()):
        st = type_stats[etype]
        count = st["count"]
        avg_d = np.mean(st["durations"]) if count else 0.0
        max_d = np.max(st["durations"]) if count else 0.0
        avg_c = np.mean(st["confidences"]) if count else 0.0
        lines.append(f"| {etype} | {count} | {avg_d:.2f} | {max_d:.2f} | {avg_c:.4f} |")

    # Table 2: Sample Reasoning Explanations
    lines.append("\n## Sample Event Explanations\n")
    lines.append("| Event ID | Event Type | Confidence | Explanation |")
    lines.append("|---|---|---|---|")
    for e in all_events[:10]:  # Show top 10 sample explanations
        lines.append(f"| {e.event_id} | {e.event_type} | {e.confidence:.0%} | {e.explanation} |")

    lines.append("\n---\n*Report automatically generated by Behaviour Reasoning Engine Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

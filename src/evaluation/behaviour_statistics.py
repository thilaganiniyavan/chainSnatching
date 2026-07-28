"""Behaviour Statistics Framework for AI Forensic Search.

Extends the evaluation framework with behaviour-level metrics:
- Behaviour frequencies (count per primitive type)
- Average interaction duration (frames + seconds)
- Behaviour transition matrix
- Interaction counts (by state)
- Average interaction confidence
- Distance statistics per interaction

Outputs:
- behaviour_statistics.csv — per-video behaviour metrics
- behaviour_report.md  — formatted research report with tables
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from src.core.models.interaction import Interaction, InteractionState
from src.core.models.behaviour_primitive import BehaviourPrimitive


# ======================================================================
# Collector
# ======================================================================

class BehaviourStatisticsCollector:
    """Collects behaviour-level metrics for a single video run.

    Args:
        video_name: Identifier for the video being processed.
        fps: Video frame rate for time-based metrics.
    """

    def __init__(self, video_name: str = "", fps: float = 30.0) -> None:
        self.video_name = video_name
        self.fps = fps if fps > 0 else 30.0

        # Behaviour frequencies
        self.behaviour_counts: Dict[str, int] = defaultdict(int)
        self.behaviour_confidence_sums: Dict[str, float] = defaultdict(float)

        # Interaction metrics
        self.interaction_durations: list[int] = []
        self.interaction_confidences: list[float] = []
        self.interaction_min_distances: list[float] = []
        self.interaction_state_counts: Dict[str, int] = defaultdict(int)

        # Transition matrix: (from_behaviour, to_behaviour) -> count
        self._transition_counts: Dict[tuple[str, str], int] = defaultdict(int)
        # Tracks the last behaviour per interaction for transition computation
        self._last_behaviour: Dict[str, str] = {}

        # Total frames processed
        self.frames_processed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        interactions: list[Interaction],
        behaviours: list[BehaviourPrimitive],
    ) -> None:
        """Record metrics from one frame's worth of data.

        Args:
            interactions: All interactions this frame (any state).
            behaviours: Behaviours detected this frame.
        """
        self.frames_processed += 1

        # Behaviour frequencies
        for bp in behaviours:
            self.behaviour_counts[bp.primitive_type] += 1
            self.behaviour_confidence_sums[bp.primitive_type] += bp.confidence

            # Transition tracking
            prev = self._last_behaviour.get(bp.interaction_id)
            if prev is not None and prev != bp.primitive_type:
                self._transition_counts[(prev, bp.primitive_type)] += 1
            self._last_behaviour[bp.interaction_id] = bp.primitive_type

    def record_completed_interaction(self, interaction: Interaction) -> None:
        """Record summary metrics for a completed interaction.

        Should be called once per interaction when it reaches ENDED/ARCHIVED.
        """
        self.interaction_durations.append(interaction.duration)
        self.interaction_confidences.append(interaction.interaction_confidence)
        self.interaction_min_distances.append(interaction.min_distance)
        self.interaction_state_counts[interaction.state.value] += 1

    def finalize(self) -> Dict[str, Any]:
        """Aggregate all collected metrics into a summary dictionary."""

        # Behaviour frequency table
        behaviour_freq: Dict[str, Any] = {}
        for btype, count in sorted(self.behaviour_counts.items()):
            avg_conf = (
                self.behaviour_confidence_sums[btype] / count if count > 0 else 0.0
            )
            behaviour_freq[btype] = {
                "count": count,
                "average_confidence": round(avg_conf, 4),
            }

        # Interaction duration stats
        n_interactions = len(self.interaction_durations)
        avg_duration_frames = (
            sum(self.interaction_durations) / n_interactions
            if n_interactions > 0
            else 0.0
        )
        max_duration_frames = (
            max(self.interaction_durations) if n_interactions > 0 else 0
        )
        min_duration_frames = (
            min(self.interaction_durations) if n_interactions > 0 else 0
        )

        # Interaction confidence stats
        avg_confidence = (
            sum(self.interaction_confidences) / n_interactions
            if n_interactions > 0
            else 0.0
        )

        # Distance stats
        avg_min_dist = (
            sum(self.interaction_min_distances) / n_interactions
            if n_interactions > 0
            else 0.0
        )

        # Transition matrix
        transition_matrix: Dict[str, Dict[str, int]] = {}
        for (from_b, to_b), count in sorted(self._transition_counts.items()):
            if from_b not in transition_matrix:
                transition_matrix[from_b] = {}
            transition_matrix[from_b][to_b] = count

        return {
            "video_name": self.video_name,
            "frames_processed": self.frames_processed,
            "behaviour_frequencies": behaviour_freq,
            "interaction_statistics": {
                "total_interactions": n_interactions,
                "state_counts": dict(self.interaction_state_counts),
                "avg_duration_frames": round(avg_duration_frames, 2),
                "avg_duration_seconds": round(avg_duration_frames / self.fps, 3),
                "max_duration_frames": max_duration_frames,
                "min_duration_frames": min_duration_frames,
                "avg_confidence": round(avg_confidence, 4),
                "avg_min_distance_px": round(avg_min_dist, 2),
            },
            "transition_matrix": transition_matrix,
        }


# ======================================================================
# File outputs
# ======================================================================

def save_behaviour_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save behaviour statistics to JSON and CSV.

    Args:
        all_stats: List of finalized statistics dictionaries (one per video).
        output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- JSON ----
    json_path = os.path.join(output_dir, "behaviour_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    # ---- CSV ----
    csv_path = os.path.join(output_dir, "behaviour_statistics.csv")
    if not all_stats:
        return

    # Collect all behaviour types across all videos for column headers
    all_btypes: set[str] = set()
    for st in all_stats:
        all_btypes.update(st.get("behaviour_frequencies", {}).keys())
    sorted_btypes = sorted(all_btypes)

    fieldnames = [
        "video_name",
        "frames_processed",
        "total_interactions",
        "avg_duration_frames",
        "avg_duration_seconds",
        "max_duration_frames",
        "avg_confidence",
        "avg_min_distance_px",
    ] + [f"behaviour_{bt}" for bt in sorted_btypes]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            row: Dict[str, Any] = {
                "video_name": st.get("video_name", ""),
                "frames_processed": st.get("frames_processed", 0),
                "total_interactions": st["interaction_statistics"]["total_interactions"],
                "avg_duration_frames": st["interaction_statistics"]["avg_duration_frames"],
                "avg_duration_seconds": st["interaction_statistics"]["avg_duration_seconds"],
                "max_duration_frames": st["interaction_statistics"]["max_duration_frames"],
                "avg_confidence": st["interaction_statistics"]["avg_confidence"],
                "avg_min_distance_px": st["interaction_statistics"]["avg_min_distance_px"],
            }
            for bt in sorted_btypes:
                freq = st.get("behaviour_frequencies", {}).get(bt, {})
                row[f"behaviour_{bt}"] = freq.get("count", 0) if isinstance(freq, dict) else 0
            writer.writerow(row)


def generate_behaviour_report(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate behaviour_report.md with tables and analysis.

    Args:
        all_stats: List of finalized statistics dictionaries.
        output_dir: Directory to write the report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "behaviour_report.md")

    lines: list[str] = []
    lines.append("# Behaviour Intelligence Layer — Evaluation Report\n")
    lines.append(f"**Videos analysed:** {len(all_stats)}\n")

    if not all_stats:
        lines.append("No statistics collected.\n")
        _write_report(report_path, lines)
        return

    # ---- Summary Table ----
    lines.append("## Interaction Summary\n")
    lines.append("| Video | Interactions | Avg Duration (f) | Avg Duration (s) | Max Duration (f) | Avg Confidence | Avg Min Dist (px) |")
    lines.append("|---|---|---|---|---|---|---|")

    for st in all_stats:
        ist = st["interaction_statistics"]
        lines.append(
            f"| {st.get('video_name', '—')} "
            f"| {ist['total_interactions']} "
            f"| {ist['avg_duration_frames']} "
            f"| {ist['avg_duration_seconds']} "
            f"| {ist['max_duration_frames']} "
            f"| {ist['avg_confidence']:.4f} "
            f"| {ist['avg_min_distance_px']:.1f} |"
        )

    # ---- Behaviour Frequencies ----
    lines.append("\n## Behaviour Frequencies\n")

    # Aggregate across all videos
    agg_freq: Dict[str, int] = defaultdict(int)
    agg_conf: Dict[str, float] = defaultdict(float)
    agg_count_conf: Dict[str, int] = defaultdict(int)
    for st in all_stats:
        for bt, info in st.get("behaviour_frequencies", {}).items():
            count = info.get("count", 0)
            agg_freq[bt] += count
            agg_conf[bt] += info.get("average_confidence", 0.0) * count
            agg_count_conf[bt] += count

    lines.append("| Behaviour | Total Count | Avg Confidence |")
    lines.append("|---|---|---|")
    for bt in sorted(agg_freq.keys()):
        avg_c = agg_conf[bt] / agg_count_conf[bt] if agg_count_conf[bt] > 0 else 0.0
        lines.append(f"| {bt} | {agg_freq[bt]} | {avg_c:.4f} |")

    # ---- Transition Matrix ----
    lines.append("\n## Behaviour Transition Matrix\n")
    lines.append("Rows are *from* behaviour, columns are *to* behaviour.\n")

    # Aggregate transitions
    agg_trans: Dict[tuple[str, str], int] = defaultdict(int)
    trans_types: set[str] = set()
    for st in all_stats:
        for from_b, targets in st.get("transition_matrix", {}).items():
            trans_types.add(from_b)
            for to_b, count in targets.items():
                trans_types.add(to_b)
                agg_trans[(from_b, to_b)] += count

    sorted_types = sorted(trans_types)
    if sorted_types:
        header = "| From \\ To | " + " | ".join(sorted_types) + " |"
        separator = "|---|" + "|".join(["---"] * len(sorted_types)) + "|"
        lines.append(header)
        lines.append(separator)
        for from_b in sorted_types:
            row_vals = [str(agg_trans.get((from_b, to_b), 0)) for to_b in sorted_types]
            lines.append(f"| {from_b} | " + " | ".join(row_vals) + " |")
    else:
        lines.append("*No transitions recorded.*\n")

    # ---- Per-Video Detail ----
    if len(all_stats) > 1:
        lines.append("\n## Per-Video Detail\n")
        for st in all_stats:
            lines.append(f"### {st.get('video_name', 'Unknown')}\n")
            ist = st["interaction_statistics"]
            lines.append(f"- Frames processed: {st.get('frames_processed', 0)}")
            lines.append(f"- Total interactions: {ist['total_interactions']}")
            lines.append(f"- Average duration: {ist['avg_duration_frames']} frames ({ist['avg_duration_seconds']}s)")
            lines.append(f"- State distribution: {ist.get('state_counts', {})}")
            lines.append("")

    lines.append("\n---\n*Report generated by the Behaviour Statistics Framework.*\n")
    _write_report(report_path, lines)


def _write_report(path: str, lines: list[str]) -> None:
    """Write lines to a markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

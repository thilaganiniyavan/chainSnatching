"""Behaviour Graph Research Evaluation & Statistics Framework.

Measures:
- Pattern frequencies & average pattern durations
- Transition frequencies & pairwise transition probabilities
- Pattern confidence distribution
- Graph depth & average graph complexity (node counts, edge counts, branching factor)
- Most common transition paths

Outputs:
- pattern_statistics.csv
- behaviour_graph_report.md
- Publication-quality figures:
  - pattern_frequency.png
  - transition_heatmap.png
  - graph_complexity_histogram.png
  - sample_behaviour_graph.png
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

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode, TransitionEdge
from src.behavior.graph_visualizer import GraphDiagramExporter


class GraphStatisticsCollector:
    """Collects research metrics for Behaviour Graphs across single or multiple video executions."""

    def __init__(self, video_name: str = "", fps: float = 30.0) -> None:
        self.video_name = video_name
        self.fps = fps if fps > 0 else 30.0
        self.graphs: list[BehaviourGraph] = []

    def record_graphs(self, graphs: list[BehaviourGraph]) -> None:
        """Record a list of BehaviourGraph objects."""
        self.graphs.extend(graphs)

    def finalize(self) -> Dict[str, Any]:
        """Aggregate research evaluation metrics across all recorded graphs."""
        total_graphs = len(self.graphs)
        if total_graphs == 0:
            return {
                "video_name": self.video_name,
                "total_graphs": 0,
                "summary": {},
            }

        # 1. Pattern frequencies and duration stats
        pattern_counts: Dict[str, int] = defaultdict(int)
        pattern_durations: Dict[str, list[float]] = defaultdict(list)
        pattern_confidences: Dict[str, list[float]] = defaultdict(list)

        # 2. Graph complexity metrics
        graph_depths: list[int] = []
        node_counts: list[int] = []
        edge_counts: list[int] = []
        branching_factors: list[float] = []

        # 3. Transition sequences and path frequencies
        transition_counts: Dict[tuple[str, str], int] = defaultdict(int)
        path_sequences: Dict[tuple[str, ...], int] = defaultdict(int)

        for graph in self.graphs:
            nodes = graph.nodes
            edges = graph.edges

            n_cnt = len(nodes)
            e_cnt = len(edges)

            node_counts.append(n_cnt)
            edge_counts.append(e_cnt)
            graph_depths.append(n_cnt)  # Sequential temporal depth

            branch_factor = e_cnt / max(1, n_cnt - 1) if n_cnt > 1 else 0.0
            branching_factors.append(branch_factor)

            seq = tuple(node.pattern_type for node in nodes)
            if seq:
                path_sequences[seq] += 1

            for node in nodes:
                ptype = node.pattern_type
                pattern_counts[ptype] += 1
                pattern_durations[ptype].append(node.duration_seconds)
                pattern_confidences[ptype].append(node.confidence)

            for edge in edges:
                pair = (edge.from_pattern_type, edge.to_pattern_type)
                transition_counts[pair] += 1

        # Summary per pattern type
        pattern_summary: Dict[str, Any] = {}
        for ptype, count in pattern_counts.items():
            durs = pattern_durations[ptype]
            confs = pattern_confidences[ptype]
            pattern_summary[ptype] = {
                "count": count,
                "avg_duration_seconds": round(float(np.mean(durs)), 3) if durs else 0.0,
                "max_duration_seconds": round(float(np.max(durs)), 3) if durs else 0.0,
                "avg_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            }

        # Transition matrix dictionary
        trans_matrix: Dict[str, Dict[str, int]] = {}
        for (from_p, to_p), cnt in transition_counts.items():
            if from_p not in trans_matrix:
                trans_matrix[from_p] = {}
            trans_matrix[from_p][to_p] = cnt

        # Top 5 most common transition paths
        top_paths = [
            {"path": " -> ".join(seq), "count": cnt}
            for seq, cnt in sorted(path_sequences.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return {
            "video_name": self.video_name,
            "total_graphs": total_graphs,
            "complexity": {
                "avg_node_count": round(float(np.mean(node_counts)), 2) if node_counts else 0.0,
                "max_node_count": int(np.max(node_counts)) if node_counts else 0,
                "avg_edge_count": round(float(np.mean(edge_counts)), 2) if edge_counts else 0.0,
                "avg_graph_depth": round(float(np.mean(graph_depths)), 2) if graph_depths else 0.0,
                "avg_branching_factor": round(float(np.mean(branching_factors)), 3) if branching_factors else 0.0,
            },
            "pattern_summary": pattern_summary,
            "transition_matrix": trans_matrix,
            "top_transition_paths": top_paths,
        }


def save_pattern_statistics(
    all_stats: List[Dict[str, Any]], output_dir: str
) -> None:
    """Save pattern statistics to pattern_statistics.csv and pattern_statistics.json."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "pattern_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "pattern_statistics.csv")
    fieldnames = [
        "video_name",
        "total_graphs",
        "avg_node_count",
        "avg_edge_count",
        "avg_graph_depth",
        "avg_branching_factor",
        "approach_pattern_count",
        "follow_pattern_count",
        "co_travel_pattern_count",
        "proximity_pattern_count",
        "interaction_pattern_count",
        "stop_pattern_count",
        "lingering_pattern_count",
        "separation_pattern_count",
        "escape_pattern_count",
        "divergence_pattern_count",
        "waiting_pattern_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for st in all_stats:
            comp = st.get("complexity", {})
            psum = st.get("pattern_summary", {})
            writer.writerow(
                {
                    "video_name": st.get("video_name", ""),
                    "total_graphs": st.get("total_graphs", 0),
                    "avg_node_count": comp.get("avg_node_count", 0.0),
                    "avg_edge_count": comp.get("avg_edge_count", 0.0),
                    "avg_graph_depth": comp.get("avg_graph_depth", 0.0),
                    "avg_branching_factor": comp.get("avg_branching_factor", 0.0),
                    "approach_pattern_count": psum.get("APPROACH_PATTERN", {}).get("count", 0),
                    "follow_pattern_count": psum.get("FOLLOW_PATTERN", {}).get("count", 0),
                    "co_travel_pattern_count": psum.get("CO_TRAVEL_PATTERN", {}).get("count", 0),
                    "proximity_pattern_count": psum.get("PROXIMITY_PATTERN", {}).get("count", 0),
                    "interaction_pattern_count": psum.get("INTERACTION_PATTERN", {}).get("count", 0),
                    "stop_pattern_count": psum.get("STOP_PATTERN", {}).get("count", 0),
                    "lingering_pattern_count": psum.get("LINGERING_PATTERN", {}).get("count", 0),
                    "separation_pattern_count": psum.get("SEPARATION_PATTERN", {}).get("count", 0),
                    "escape_pattern_count": psum.get("ESCAPE_PATTERN", {}).get("count", 0),
                    "divergence_pattern_count": psum.get("DIVERGENCE_PATTERN", {}).get("count", 0),
                    "waiting_pattern_count": psum.get("WAITING_PATTERN", {}).get("count", 0),
                }
            )


def generate_publication_figures(
    all_graphs: List[BehaviourGraph], output_dir: str
) -> None:
    """Generate publication-quality research figures:

    - pattern_frequency.png
    - transition_heatmap.png
    - graph_complexity_histogram.png
    - sample_behaviour_graph.png
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_graphs:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Pattern Frequency Chart
    pattern_counts: Dict[str, int] = defaultdict(int)
    for g in all_graphs:
        for node in g.nodes:
            pattern_counts[node.pattern_type] += 1

    ptypes = sorted(pattern_counts.keys())
    counts = [pattern_counts[t] for t in ptypes]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(ptypes, counts, color="#2b5c8f", width=0.55)
    plt.title("Behaviour Pattern Occurrences Across Graphs", fontsize=14, fontweight="bold")
    plt.xlabel("Pattern Type", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, str(int(yval)), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pattern_frequency.png"), dpi=300)
    plt.close()

    # 2. Graph Complexity Histogram (Node Counts per Graph)
    node_counts = [len(g.nodes) for g in all_graphs]
    plt.figure(figsize=(8, 5))
    plt.hist(node_counts, bins=range(1, max(max(node_counts, default=1) + 2, 6)), color="#02818a", edgecolor="black", align="left")
    plt.title("Graph Complexity Distribution (Node Count Depth)", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Pattern Nodes per Graph", fontsize=12)
    plt.ylabel("Graph Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_complexity_histogram.png"), dpi=300)
    plt.close()

    # 3. Pairwise Transition Heatmap
    trans_counts: Dict[tuple[str, str], int] = defaultdict(int)
    all_p_set: set[str] = set()
    for g in all_graphs:
        for edge in g.edges:
            all_p_set.add(edge.from_pattern_type)
            all_p_set.add(edge.to_pattern_type)
            trans_counts[(edge.from_pattern_type, edge.to_pattern_type)] += 1

    sorted_p = sorted(all_p_set)
    if sorted_p:
        matrix_data = np.zeros((len(sorted_p), len(sorted_p)), dtype=int)
        for i, p1 in enumerate(sorted_p):
            for j, p2 in enumerate(sorted_p):
                matrix_data[i, j] = trans_counts.get((p1, p2), 0)

        plt.figure(figsize=(9, 7))
        plt.imshow(matrix_data, cmap="YlGnBu")
        plt.title("Pattern Transition Matrix Heatmap", fontsize=14, fontweight="bold")
        plt.xticks(range(len(sorted_p)), sorted_p, rotation=45, ha="right", fontsize=9)
        plt.yticks(range(len(sorted_p)), sorted_p, fontsize=9)
        plt.xlabel("Target Pattern (To)", fontsize=11)
        plt.ylabel("Source Pattern (From)", fontsize=11)
        plt.colorbar(label="Transition Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "transition_heatmap.png"), dpi=300)
        plt.close()

    # 4. Export sample diagram for the largest graph
    largest_graph = max(all_graphs, key=lambda g: len(g.nodes))
    sample_path = os.path.join(output_dir, "sample_behaviour_graph.png")
    GraphDiagramExporter.export_diagram(largest_graph, sample_path)


def generate_behaviour_graph_report(
    all_stats: List[Dict[str, Any]],
    all_graphs: List[BehaviourGraph],
    output_dir: str,
) -> None:
    """Generate behaviour_graph_report.md containing research evaluation analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "behaviour_graph_report.md")

    lines: list[str] = []
    lines.append("# Behaviour Graph Reasoning Engine — Research Evaluation Report\n")
    lines.append(f"**Total Video Runs:** {len(all_stats)}")
    lines.append(f"**Total Behaviour Graphs Generated:** {len(all_graphs)}\n")

    if not all_graphs:
        lines.append("No behaviour graphs generated.\n")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # Section 1: Graph Complexity Summary
    lines.append("## Graph Complexity Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    node_counts = [len(g.nodes) for g in all_graphs]
    edge_counts = [len(g.edges) for g in all_graphs]
    lines.append(f"| Average Nodes per Graph | {np.mean(node_counts):.2f} |")
    lines.append(f"| Maximum Graph Depth | {np.max(node_counts)} |")
    lines.append(f"| Average Edges per Graph | {np.mean(edge_counts):.2f} |")
    lines.append(f"| Average Branching Factor | {np.mean([len(g.edges)/max(1, len(g.nodes)-1) for g in all_graphs]):.3f} |\n")

    # Section 2: Behaviour Pattern Summary
    pattern_counts: Dict[str, int] = defaultdict(int)
    pattern_durs: Dict[str, list[float]] = defaultdict(list)
    pattern_confs: Dict[str, list[float]] = defaultdict(list)

    for g in all_graphs:
        for n in g.nodes:
            pattern_counts[n.pattern_type] += 1
            pattern_durs[n.pattern_type].append(n.duration_seconds)
            pattern_confs[n.pattern_type].append(n.confidence)

    lines.append("## Behaviour Pattern Frequencies & Durations\n")
    lines.append("| Pattern Type | Total Count | Avg Duration (s) | Max Duration (s) | Avg Confidence |")
    lines.append("|---|---|---|---|---|")

    for ptype in sorted(pattern_counts.keys()):
        cnt = pattern_counts[ptype]
        avg_d = np.mean(pattern_durs[ptype]) if cnt else 0.0
        max_d = np.max(pattern_durs[ptype]) if cnt else 0.0
        avg_c = np.mean(pattern_confs[ptype]) if cnt else 0.0
        lines.append(f"| {ptype} | {cnt} | {avg_d:.2f} | {max_d:.2f} | {avg_c:.4f} |")

    # Section 3: Common Transition Paths
    path_counts: Dict[str, int] = defaultdict(int)
    for g in all_graphs:
        path = " -> ".join(n.pattern_type for n in g.nodes)
        if path:
            path_counts[path] += 1

    lines.append("\n## Most Common Transition Paths\n")
    lines.append("| Rank | Pattern Transition Path | Count |")
    lines.append("|---|---|---|")
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for idx, (path_str, cnt) in enumerate(sorted_paths, start=1):
        lines.append(f"| {idx} | `{path_str}` | {cnt} |")

    lines.append("\n---\n*Report generated by the Behaviour Graph Research Evaluation Framework.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

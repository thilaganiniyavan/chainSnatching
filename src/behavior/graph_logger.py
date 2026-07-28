"""Graph Logger & Data Storage — structured JSON and CSV dataset exporter.

Generates:
- ``behaviour_graph.json``
- ``behaviour_patterns.csv``
- ``transition_matrix.csv``
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode, TransitionEdge


class GraphLogger:
    """Logs and exports Behaviour Graphs to JSON and CSV formats."""

    def __init__(self) -> None:
        self._graphs: list[BehaviourGraph] = []

    def log_graph(self, graph: BehaviourGraph) -> None:
        """Store a BehaviourGraph for export."""
        self._graphs.append(graph)

    def log_graphs(self, graphs: list[BehaviourGraph]) -> None:
        """Store multiple BehaviourGraphs for export."""
        for g in graphs:
            self.log_graph(g)

    def export_json(self, output_path: str) -> None:
        """Export all logged graphs to behaviour_graph.json."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        graph_dicts = [self._graph_to_dict(g) for g in self._graphs]

        # Aggregate total pattern sequence across all graphs
        all_pattern_nodes = [node for g in self._graphs for node in g.nodes]
        pattern_type_counts: dict[str, int] = {}
        for node in all_pattern_nodes:
            pattern_type_counts[node.pattern_type] = pattern_type_counts.get(node.pattern_type, 0) + 1

        payload = {
            "graphs": graph_dicts,
            "summary": {
                "total_graphs": len(self._graphs),
                "total_pattern_nodes": len(all_pattern_nodes),
                "pattern_type_counts": pattern_type_counts,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def export_patterns_csv(self, output_path: str) -> None:
        """Export all pattern nodes across all graphs to behaviour_patterns.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "graph_id",
            "interaction_id",
            "pattern_id",
            "pattern_type",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_seconds",
            "confidence",
            "supporting_primitives",
            "min_distance_px",
            "avg_distance_px",
            "peak_relative_velocity",
            "peak_relative_acceleration",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for g in self._graphs:
                for node in g.nodes:
                    writer.writerow(
                        {
                            "graph_id": g.graph_id,
                            "interaction_id": g.interaction_id,
                            "pattern_id": node.pattern_id,
                            "pattern_type": node.pattern_type,
                            "start_frame": node.start_frame,
                            "end_frame": node.end_frame,
                            "duration_frames": node.duration_frames,
                            "duration_seconds": node.duration_seconds,
                            "confidence": node.confidence,
                            "supporting_primitives": " -> ".join(node.supporting_primitives),
                            "min_distance_px": node.supporting_spatial.get("min_distance", 0.0),
                            "avg_distance_px": node.supporting_spatial.get("avg_distance", 0.0),
                            "peak_relative_velocity": node.supporting_motion.get("peak_relative_velocity", 0.0),
                            "peak_relative_acceleration": node.supporting_motion.get("peak_relative_acceleration", 0.0),
                        }
                    )

    def export_transition_matrix_csv(self, output_path: str) -> None:
        """Export pairwise pattern transition matrix to transition_matrix.csv."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Collect pairwise transition counts
        counts: dict[tuple[str, str], int] = {}
        all_patterns: set[str] = set()

        for g in self._graphs:
            for edge in g.edges:
                all_patterns.add(edge.from_pattern_type)
                all_patterns.add(edge.to_pattern_type)
                pair = (edge.from_pattern_type, edge.to_pattern_type)
                counts[pair] = counts.get(pair, 0) + 1

        sorted_patterns = sorted(all_patterns)
        fieldnames = ["from_pattern"] + sorted_patterns

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for from_p in sorted_patterns:
                row: dict[str, Any] = {"from_pattern": from_p}
                for to_p in sorted_patterns:
                    row[to_p] = counts.get((from_p, to_p), 0)
                writer.writerow(row)

    def export_all(
        self,
        json_path: str,
        patterns_csv_path: str,
        transition_csv_path: str,
    ) -> None:
        """Export JSON, patterns CSV, and transition matrix CSV."""
        self.export_json(json_path)
        self.export_patterns_csv(patterns_csv_path)
        self.export_transition_matrix_csv(transition_csv_path)

    def get_graphs(self) -> list[BehaviourGraph]:
        """Return all logged graphs."""
        return list(self._graphs)

    def clear(self) -> None:
        """Clear internal graph log storage."""
        self._graphs.clear()

    @staticmethod
    def _graph_to_dict(graph: BehaviourGraph) -> dict[str, Any]:
        """Serialise a BehaviourGraph instance to a clean dictionary."""
        return {
            "graph_id": graph.graph_id,
            "interaction_id": graph.interaction_id,
            "person_track_id": graph.person_track_id,
            "vehicle_track_id": graph.vehicle_track_id,
            "start_frame": graph.start_frame,
            "end_frame": graph.end_frame,
            "is_active": graph.is_active,
            "nodes": [
                {
                    "pattern_id": n.pattern_id,
                    "pattern_type": n.pattern_type,
                    "start_frame": n.start_frame,
                    "end_frame": n.end_frame,
                    "duration_frames": n.duration_frames,
                    "duration_seconds": n.duration_seconds,
                    "confidence": n.confidence,
                    "supporting_primitives": n.supporting_primitives,
                    "supporting_motion": n.supporting_motion,
                    "supporting_spatial": n.supporting_spatial,
                    "extensible_evidence": n.extensible_evidence,
                }
                for n in graph.nodes
            ],
            "edges": [
                {
                    "source_pattern_id": e.source_pattern_id,
                    "target_pattern_id": e.target_pattern_id,
                    "from_pattern_type": e.from_pattern_type,
                    "to_pattern_type": e.to_pattern_type,
                    "transition_frame": e.transition_frame,
                    "timestamp": e.timestamp,
                    "transition_confidence": e.transition_confidence,
                    "transition_condition": e.transition_condition,
                }
                for e in graph.edges
            ],
            "pattern_sequence": [n.pattern_type for n in graph.nodes],
            "extensible_metadata": graph.extensible_metadata,
        }

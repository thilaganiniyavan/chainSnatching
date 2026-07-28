"""Graph Visualizer — video overlay & NetworkX graph diagram exporter.

Provides:
1. ``OverlayVisualizer``: Renders OpenCV video HUD showing current pattern,
   graph depth, node count, edge count, duration, and confidence.
2. ``GraphDiagramExporter``: Renders and saves directed graph structure plots (.png)
   using NetworkX + Matplotlib (with optional PyDot/Graphviz fallback) for debugging
   and research inspection.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode
from src.core.models.track import Track


# Pattern Type -> Colour Mapping (BGR for OpenCV, RGB for NetworkX)
_PATTERN_COLOURS_CV: dict[str, tuple[int, int, int]] = {
    "APPROACH_PATTERN": (255, 150, 0),       # Blue-ish
    "FOLLOW_PATTERN": (0, 165, 255),        # Orange
    "CO_TRAVEL_PATTERN": (0, 200, 200),      # Yellow-Green
    "PROXIMITY_PATTERN": (0, 255, 255),      # Yellow
    "INTERACTION_PATTERN": (0, 120, 255),    # Red-Orange
    "STOP_PATTERN": (200, 200, 0),           # Teal
    "LINGERING_PATTERN": (180, 180, 0),      # Dark Teal
    "SEPARATION_PATTERN": (255, 100, 100),   # Light Blue
    "ESCAPE_PATTERN": (0, 0, 255),           # Red
    "DIVERGENCE_PATTERN": (150, 0, 200),     # Purple
    "WAITING_PATTERN": (0, 180, 255),        # Amber
}

_PATTERN_COLOURS_NX: dict[str, str] = {
    "APPROACH_PATTERN": "#1f77b4",
    "FOLLOW_PATTERN": "#ff7f0e",
    "CO_TRAVEL_PATTERN": "#2ca02c",
    "PROXIMITY_PATTERN": "#bcbd22",
    "INTERACTION_PATTERN": "#d62728",
    "STOP_PATTERN": "#17becf",
    "LINGERING_PATTERN": "#8c564b",
    "SEPARATION_PATTERN": "#aec7e8",
    "ESCAPE_PATTERN": "#e377c2",
    "DIVERGENCE_PATTERN": "#9467bd",
    "WAITING_PATTERN": "#ffbb78",
}


class OverlayVisualizer:
    """Renders real-time Behaviour Graph status overlays on video frames.

    Args:
        font_scale: Font scale for OpenCV drawing.
        panel_alpha: Semi-transparent panel alpha.
        fps: Video FPS.
    """

    def __init__(
        self,
        font_scale: float = 0.45,
        panel_alpha: float = 0.65,
        fps: float = 30.0,
    ) -> None:
        self.font_scale = font_scale
        self.panel_alpha = panel_alpha
        self.fps = fps if fps > 0 else 30.0
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        graphs: list[BehaviourGraph],
        tracks: list[Track],
    ) -> np.ndarray:
        """Annotate *frame* with current graph state & active pattern HUD panels."""
        viz = frame.copy()

        track_centers: dict[int, tuple[int, int]] = {
            t.tracking_id: t.center for t in tracks if t.center is not None
        }

        for graph in graphs:
            if not graph.nodes:
                continue

            active_node = graph.nodes[-1]
            p_center = track_centers.get(graph.person_track_id)
            v_center = track_centers.get(graph.vehicle_track_id)

            if p_center is None or v_center is None:
                continue

            colour = _PATTERN_COLOURS_CV.get(active_node.pattern_type, (255, 255, 255))
            self._draw_graph_hud(viz, graph, active_node, p_center, v_center, colour)

        return viz

    def _draw_graph_hud(
        self,
        frame: np.ndarray,
        graph: BehaviourGraph,
        active_node: PatternNode,
        p_center: tuple[int, int],
        v_center: tuple[int, int],
        colour: tuple[int, int, int],
    ) -> None:
        """Draw graph status panel near vehicle center."""
        depth = len(graph.nodes)
        edge_cnt = len(graph.edges)

        lines = [
            f"GRAPH: {graph.graph_id}",
            f"Pattern: {active_node.pattern_type}",
            f"Dur: {active_node.duration_frames}f ({active_node.duration_seconds:.1f}s) | Conf: {active_node.confidence:.0%}",
            f"Graph State: Nodes={depth} | Edges={edge_cnt} | Depth={depth}",
        ]

        padding = 6
        line_height = int(18 * self.font_scale / 0.45)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self._font, self.font_scale, 1)
            max_w = max(max_w, tw)

        panel_w = max_w + 2 * padding
        panel_h = line_height * len(lines) + 2 * padding

        anchor_x = v_center[0] + 10
        anchor_y = v_center[1] + 10

        h, w = frame.shape[:2]
        if anchor_x + panel_w > w:
            anchor_x = max(5, w - panel_w - 5)
        if anchor_y + panel_h > h:
            anchor_y = max(5, h - panel_h - 5)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            (25, 25, 25),
            -1,
        )
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1 - self.panel_alpha, 0, frame)

        cv2.rectangle(
            frame,
            (anchor_x, anchor_y),
            (anchor_x + panel_w, anchor_y + panel_h),
            colour,
            2,
        )

        for i, line in enumerate(lines):
            ty = anchor_y + padding + line_height * (i + 1) - 3
            line_colour = colour if i < 2 else (220, 220, 220)
            cv2.putText(
                frame,
                line,
                (anchor_x + padding, ty),
                self._font,
                self.font_scale,
                line_colour,
                1,
            )


class GraphDiagramExporter:
    """Exports directed graph structure diagrams using NetworkX and Matplotlib."""

    @staticmethod
    def export_diagram(graph: BehaviourGraph, output_path: str) -> None:
        """Render and save a directed graph diagram for a single BehaviourGraph."""
        if not graph.nodes:
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        G = nx.DiGraph()

        # Add nodes with pattern types and labels
        node_colors = []
        labels = {}

        for node in graph.nodes:
            label = f"{node.pattern_type}\n({node.duration_seconds:.1f}s, {node.confidence:.0%})"
            G.add_node(node.pattern_id, label=label, pattern_type=node.pattern_type)
            color = _PATTERN_COLOURS_NX.get(node.pattern_type, "#7f7f7f")
            node_colors.append(color)
            labels[node.pattern_id] = label

        # Add directed edges
        for edge in graph.edges:
            G.add_edge(
                edge.source_pattern_id,
                edge.target_pattern_id,
                label=f"{edge.transition_confidence:.0%}",
            )

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42) if len(G) > 1 else {graph.nodes[0].pattern_id: (0, 0)}

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

        if graph.edges:
            nx.draw_networkx_edges(
                G, pos, arrowstyle="->", arrowsize=15, edge_color="#555555", width=2
            )
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title(
            f"Behaviour Graph — {graph.graph_id}\n"
            f"(Interaction {graph.interaction_id} | Person {graph.person_track_id} <-> Vehicle {graph.vehicle_track_id})",
            fontsize=12,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def export_all_diagrams(graphs: list[BehaviourGraph], output_dir: str) -> None:
        """Export graph diagrams for all provided graphs."""
        os.makedirs(output_dir, exist_ok=True)
        for graph in graphs:
            path = os.path.join(output_dir, f"graph_{graph.interaction_id}.png")
            GraphDiagramExporter.export_diagram(graph, path)

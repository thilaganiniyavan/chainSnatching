"""Behaviour Graph Engine — directed state-transition graph builder.

Transforms chronological behaviour primitives and interaction history into
reusable directed Behaviour Graphs.

Nodes represent higher-level Behaviour Patterns. Edges represent temporal
state transitions between consecutive patterns.

Provides the full requested API suite:
- create_graph()
- update_graph()
- finalize_graph()
- get_graph()
- get_active_graphs()
- get_completed_graphs()
- find_patterns()
- get_transition_statistics()

Does NOT classify chain-snatching or suspicious intent — acts as a pure,
explainable intermediate representation for future ST-GCN / Pose / Fusion models.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models.behaviour_graph import BehaviourGraph, PatternNode, TransitionEdge
from src.core.models.behaviour_primitive import BehaviourPrimitive
from src.core.models.interaction import Interaction, InteractionState
from src.behavior.behaviour_timeline import TimelineEvent
from src.behavior.pattern_evaluator import PatternEvaluator
from src.behavior.pattern_rules import PatternConfig


class BehaviourGraphEngine:
    """Manages creation, state updates, edge creation, finalization, and querying
    of directed Behaviour Graphs for all video interactions.

    Args:
        pattern_config: Optional PatternConfig for PatternEvaluator tuning.
        fps: Video FPS for converting frame durations and timestamps.
    """

    def __init__(
        self,
        pattern_config: PatternConfig | None = None,
        fps: float = 30.0,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.evaluator = PatternEvaluator(config=pattern_config, fps=self.fps)

        # Internal graph registry: interaction_id -> BehaviourGraph
        self._graphs: dict[str, BehaviourGraph] = {}

        # Counter for unique graph IDs
        self._graph_counter: int = 0
        self._node_counter: int = 0

        # Last active pattern node type per graph to avoid duplicate adjacent nodes
        self._last_pattern_type: dict[str, str] = {}
        self._last_pattern_id: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API Suite
    # ------------------------------------------------------------------

    def create_graph(
        self,
        interaction: Interaction,
        frame_number: int,
    ) -> BehaviourGraph:
        """Initialize a new directed Behaviour Graph for a new interaction.

        Returns the newly created BehaviourGraph object.
        """
        self._graph_counter += 1
        graph_id = f"GRAPH-{interaction.interaction_id}"

        graph = BehaviourGraph(
            graph_id=graph_id,
            interaction_id=interaction.interaction_id,
            person_track_id=interaction.person_track_id,
            vehicle_track_id=interaction.vehicle_track_id,
            start_frame=frame_number,
            is_active=True,
            extensible_metadata={
                "created_at_frame": frame_number,
                "fps": self.fps,
                "skeleton_sequence": [],     # Future ST-GCN extension hook
                "fusion_features": {},       # Future Fusion extension hook
            },
        )

        self._graphs[interaction.interaction_id] = graph
        return graph

    def update_graph(
        self,
        interaction: Interaction,
        primitives: list[BehaviourPrimitive],
        timeline: list[TimelineEvent],
        frame_number: int,
    ) -> BehaviourGraph:
        """Main per-frame update call.

        Evaluates active patterns, appends new PatternNode instances when pattern
        changes occur, creates TransitionEdge instances connecting consecutive nodes,
        and updates current graph state.
        """
        iid = interaction.interaction_id

        # Auto-create graph if not yet registered
        if iid not in self._graphs:
            graph = self.create_graph(interaction, frame_number)
        else:
            graph = self._graphs[iid]

        if not graph.is_active:
            return graph

        # Evaluate active pattern nodes for current frame
        detected_patterns = self.evaluator.evaluate(
            interaction, primitives, timeline, frame_number
        )

        if not detected_patterns:
            return graph

        # Select primary pattern (highest priority then confidence)
        primary_pattern = max(detected_patterns, key=lambda p: (p.priority, p.confidence))

        last_type = self._last_pattern_type.get(iid)
        last_node_id = self._last_pattern_id.get(iid)

        # Check if the primary pattern changed or if this is the first pattern node
        if primary_pattern.pattern_type != last_type:
            self._node_counter += 1
            node_id = f"PAT-{self._node_counter:04d}"
            primary_pattern.pattern_id = node_id

            graph.nodes.append(primary_pattern)

            # If there was a previous pattern, create a TransitionEdge
            if last_node_id is not None and last_type is not None:
                edge = TransitionEdge(
                    source_pattern_id=last_node_id,
                    target_pattern_id=node_id,
                    from_pattern_type=last_type,
                    to_pattern_type=primary_pattern.pattern_type,
                    transition_frame=frame_number,
                    timestamp=round(frame_number / self.fps, 3),
                    transition_confidence=round(primary_pattern.confidence, 4),
                    transition_condition=f"{last_type} -> {primary_pattern.pattern_type}",
                )
                graph.edges.append(edge)

            self._last_pattern_type[iid] = primary_pattern.pattern_type
            self._last_pattern_id[iid] = node_id
        else:
            # Update end frame and duration of existing active node
            if graph.nodes:
                active_node = graph.nodes[-1]
                active_node.end_frame = frame_number
                active_node.duration_frames = frame_number - active_node.start_frame + 1
                active_node.duration_seconds = round(active_node.duration_frames / self.fps, 3)
                active_node.confidence = max(active_node.confidence, primary_pattern.confidence)
                active_node.supporting_motion = primary_pattern.supporting_motion
                active_node.supporting_spatial = primary_pattern.supporting_spatial

        # Auto-finalize if interaction ended
        if interaction.state in (InteractionState.ENDED, InteractionState.ARCHIVED):
            self.finalize_graph(iid, frame_number)

        return graph

    def finalize_graph(
        self,
        interaction_id: str,
        frame_number: int,
    ) -> BehaviourGraph:
        """Finalize an interaction graph when the interaction ends."""
        graph = self._graphs.get(interaction_id)
        if graph is not None and graph.is_active:
            graph.is_active = False
            graph.end_frame = frame_number
        return graph if graph is not None else self._graphs.get(interaction_id)

    def get_graph(self, interaction_id: str) -> Optional[BehaviourGraph]:
        """Return the BehaviourGraph for a given interaction ID."""
        return self._graphs.get(interaction_id)

    def get_active_graphs(self) -> list[BehaviourGraph]:
        """Return all currently active Behaviour Graphs."""
        return [g for g in self._graphs.values() if g.is_active]

    def get_completed_graphs(self) -> list[BehaviourGraph]:
        """Return all completed (finalized) Behaviour Graphs."""
        return [g for g in self._graphs.values() if not g.is_active]

    def get_all_graphs(self) -> list[BehaviourGraph]:
        """Return all graphs (active and completed)."""
        return list(self._graphs.values())

    def find_patterns(
        self,
        pattern_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[PatternNode]:
        """Search across all graphs for PatternNodes matching criteria."""
        matched: list[PatternNode] = []
        for graph in self._graphs.values():
            for node in graph.nodes:
                if node.confidence >= min_confidence:
                    if pattern_type is None or node.pattern_type == pattern_type:
                        matched.append(node)
        return matched

    def get_transition_statistics(self) -> dict[str, Any]:
        """Compute aggregated transition statistics across all managed graphs.

        Returns transition counts, transition probabilities, and average edge confidences.
        """
        counts: dict[tuple[str, str], int] = {}
        conf_sums: dict[tuple[str, str], float] = {}
        from_totals: dict[str, int] = {}

        for graph in self._graphs.values():
            for edge in graph.edges:
                key = (edge.from_pattern_type, edge.to_pattern_type)
                counts[key] = counts.get(key, 0) + 1
                conf_sums[key] = conf_sums.get(key, 0.0) + edge.transition_confidence
                from_totals[edge.from_pattern_type] = from_totals.get(edge.from_pattern_type, 0) + 1

        transitions: list[dict[str, Any]] = []
        matrix: dict[str, dict[str, float]] = {}

        for (from_t, to_t), cnt in counts.items():
            avg_conf = conf_sums[(from_t, to_t)] / cnt
            prob = cnt / from_totals[from_t] if from_totals[from_t] > 0 else 0.0

            transitions.append(
                {
                    "from_pattern": from_t,
                    "to_pattern": to_t,
                    "count": cnt,
                    "probability": round(prob, 4),
                    "average_confidence": round(avg_conf, 4),
                }
            )

            if from_t not in matrix:
                matrix[from_t] = {}
            matrix[from_t][to_t] = round(prob, 4)

        return {
            "total_transitions": sum(counts.values()),
            "unique_transition_pairs": len(counts),
            "transitions": transitions,
            "transition_matrix": matrix,
        }

    def clear(self) -> None:
        """Clear all internal registries."""
        self._graphs.clear()
        self._last_pattern_type.clear()
        self._last_pattern_id.clear()

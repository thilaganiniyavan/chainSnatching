"""Fusion Temporal Aligner — multi-modal timeline synchronization.

Aligns asynchronous frame timestamps, Behaviour Graph pattern nodes,
human action classification windows, and motion trajectory statistics into
a unified, frame-indexed evidence timeline.
"""

from __future__ import annotations

from typing import Any

from src.core.models.behaviour_graph import BehaviourGraph
from src.core.models.action_result import ActionResult


class FusionTemporalAligner:
    """Synchronizes multi-modal evidence streams across timestamps and frame indices."""

    def align_streams(
        self,
        graph: BehaviourGraph,
        action_results: list[ActionResult],
        fps: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Align Behaviour Graph nodes and Action Results into a chronological evidence timeline.

        Args:
            graph: Source BehaviourGraph object.
            action_results: List of ActionResult objects for this interaction.
            fps: Video FPS.

        Returns:
            List of synchronized timeline dictionaries:
            ``{"frame": int, "timestamp": float, "behaviour_pattern": str, "action_label": str, "confidence": float}``
        """
        timeline_dict: dict[int, dict[str, Any]] = {}

        # 1. Align Behaviour Graph pattern nodes
        for node in graph.nodes:
            f_num = getattr(node, 'frame_number', getattr(node, 'start_frame', 0))
            ts = round(f_num / max(1.0, fps), 3)

            timeline_dict[f_num] = {
                "frame": f_num,
                "timestamp": ts,
                "behaviour_pattern": node.pattern_type,
                "behaviour_confidence": node.confidence,
                "action_label": "Unknown",
                "action_confidence": 0.0,
            }

        # 2. Align Action Results
        for act in action_results:
            # Action results may span multiple frame steps
            seq_metadata = act.metadata
            frame_indices = seq_metadata.get("frame_indices", [])

            if not frame_indices:
                # Fallback to estimation from action sequence ID
                f_num = graph.start_frame
                frame_indices = [f_num]

            for f_num in frame_indices:
                ts = round(f_num / max(1.0, fps), 3)
                if f_num not in timeline_dict:
                    timeline_dict[f_num] = {
                        "frame": f_num,
                        "timestamp": ts,
                        "behaviour_pattern": "NONE",
                        "behaviour_confidence": 0.0,
                        "action_label": act.predicted_action,
                        "action_confidence": act.action_confidence,
                    }
                else:
                    timeline_dict[f_num]["action_label"] = act.predicted_action
                    timeline_dict[f_num]["action_confidence"] = act.action_confidence

        # Sort timeline by frame number
        sorted_timeline = [timeline_dict[k] for k in sorted(timeline_dict.keys())]
        return sorted_timeline

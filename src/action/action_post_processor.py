"""Action Post-Processor — post-processing for ActionResult outputs.

Implements:
- Confidence threshold filtering (replaces low confidence predictions with "Unknown")
- Top-k probability ranking
- Temporal smoothing / majority voting across consecutive sliding window clips
- Overlapping window probability aggregation
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.core.models.action_result import ActionResult


class ActionPostProcessor:
    """Post-processes raw ActionResult classifications.

    Args:
        min_confidence: Confidence threshold below which predicted action is set to "Unknown".
        top_k: Number of top probability predictions to retain.
    """

    def __init__(
        self,
        min_confidence: float = 0.40,
        top_k: int = 5,
    ) -> None:
        self.min_confidence = min_confidence
        self.top_k = top_k

    def process(self, action_result: ActionResult) -> ActionResult:
        """Apply confidence filtering and top-k ranking to *action_result*.

        Args:
            action_result: Input ActionResult object.

        Returns:
            Updated :class:`ActionResult` object.
        """
        probs = action_result.class_probabilities
        if not probs:
            action_result.predicted_action = "Unknown"
            action_result.action_confidence = 0.0
            return action_result

        # Sort predictions by probability score
        sorted_preds = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        action_result.top_k_predictions = sorted_preds[: self.top_k]

        top_label, top_conf = sorted_preds[0]

        # Apply confidence threshold fallback to "Unknown"
        if top_conf < self.min_confidence:
            action_result.predicted_action = "Unknown"
            action_result.action_confidence = round(top_conf, 4)
            action_result.metadata["fallback_triggered"] = True
            action_result.metadata["original_prediction"] = top_label
        else:
            action_result.predicted_action = top_label
            action_result.action_confidence = round(top_conf, 4)

        return action_result

    def process_batch(self, action_results: list[ActionResult]) -> list[ActionResult]:
        """Apply post-processing to a list of ActionResult objects."""
        return [self.process(res) for res in action_results]

    def aggregate_overlapping_windows(
        self,
        clip_results: list[ActionResult],
    ) -> ActionResult:
        """Aggregate predictions from multiple sliding window clips of the same interaction track."""
        if not clip_results:
            return ActionResult()

        first = clip_results[0]
        sum_probs: dict[str, float] = defaultdict(float)

        for res in clip_results:
            for cls_name, prob in res.class_probabilities.items():
                sum_probs[cls_name] += prob

        n_clips = max(1, len(clip_results))
        avg_probs = {cls_name: round(prob_sum / n_clips, 4) for cls_name, prob_sum in sum_probs.items()}

        sorted_preds = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_preds[0] if sorted_preds else ("Unknown", 0.0)

        final_label = top_label if top_conf >= self.min_confidence else "Unknown"

        return ActionResult(
            sequence_id=first.sequence_id,
            interaction_id=first.interaction_id,
            track_id=first.track_id,
            predicted_action=final_label,
            action_confidence=round(top_conf, 4),
            class_probabilities=avg_probs,
            top_k_predictions=sorted_preds[: self.top_k],
            inference_time_ms=round(sum(r.inference_time_ms for r in clip_results), 2),
            model_name=first.model_name,
            model_version=first.model_version,
            device_used=first.device_used,
            skeleton_quality=first.skeleton_quality,
            metadata={"aggregated_clip_count": n_clips},
        )

"""CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network) Framework Adapter Scaffold.

Allows future integration of CTR-GCN model without changing downstream modules.
"""

from __future__ import annotations

from src.core.models.action_result import ActionResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.action.base_recognizer import AbstractActionRecognizer
from src.action.stgcn_recognizer import STGCNRecognizer


class CTRGCNRecognizer(AbstractActionRecognizer):
    """Adapter scaffold for CTR-GCN model architecture."""

    def __init__(self, action_taxonomy: list[str] | None = None) -> None:
        super().__init__(
            backend_name="CTR-GCN",
            action_taxonomy=action_taxonomy,
        )
        self._fallback = STGCNRecognizer(action_taxonomy=action_taxonomy)

    def predict_action(self, sequence: SkeletonSequence) -> ActionResult:
        res = self._fallback.predict_action(sequence)
        res.model_name = "CTR-GCN"
        res.metadata["scaffold_adapter"] = True
        return res

    def predict_batch(self, sequences: list[SkeletonSequence]) -> list[ActionResult]:
        results = self._fallback.predict_batch(sequences)
        for r in results:
            r.model_name = "CTR-GCN"
            r.metadata["scaffold_adapter"] = True
        return results

"""MSG-3D (Multi-Scale Graph 3D Convolutional Network) Framework Adapter Scaffold.

Allows future integration of MSG-3D model without changing downstream modules.
"""

from __future__ import annotations

from src.core.models.action_result import ActionResult
from src.core.models.skeleton_sequence import SkeletonSequence
from src.action.base_recognizer import AbstractActionRecognizer
from src.action.stgcn_recognizer import STGCNRecognizer


class MSG3DRecognizer(AbstractActionRecognizer):
    """Adapter scaffold for MSG-3D model architecture."""

    def __init__(self, action_taxonomy: list[str] | None = None) -> None:
        super().__init__(
            backend_name="MSG-3D",
            action_taxonomy=action_taxonomy,
        )
        self._fallback = STGCNRecognizer(action_taxonomy=action_taxonomy)

    def predict_action(self, sequence: SkeletonSequence) -> ActionResult:
        res = self._fallback.predict_action(sequence)
        res.model_name = "MSG-3D"
        res.metadata["scaffold_adapter"] = True
        return res

    def predict_batch(self, sequences: list[SkeletonSequence]) -> list[ActionResult]:
        results = self._fallback.predict_batch(sequences)
        for r in results:
            r.model_name = "MSG-3D"
            r.metadata["scaffold_adapter"] = True
        return results

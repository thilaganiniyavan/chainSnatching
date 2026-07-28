"""Snatch Signature Engine — forensic crime signature evaluation manager.

Consumes multi-modal FusedInteraction objects, executes template matching via
:class:`SignatureMatcher`, generates forensic explanations via :class:`SignatureExplainer`,
and maintains searchable forensic signature results.

Exposes clean API suite:
- evaluate_interaction()
- evaluate_batch()
- get_signature_result()
- get_flagged_results()
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.models.fused_interaction import FusedInteraction
from src.core.models.snatch_signature_result import SnatchSignatureResult
from src.snatch.signature_config import SignatureTemplate, StandardMotorcycleSnatchSignature
from src.snatch.signature_matcher import SignatureMatcher
from src.snatch.signature_explainer import SignatureExplainer


class SnatchSignatureEngine:
    """Engine executing crime-specific forensic signature matching over FusedInteractions.

    Args:
        template: Optional custom SignatureTemplate (defaults to StandardMotorcycleSnatchSignature).
    """

    def __init__(self, template: SignatureTemplate | None = None) -> None:
        self.matcher = SignatureMatcher(template=template)
        self.explainer = SignatureExplainer()
        self._results_map: dict[str, SnatchSignatureResult] = {}

    def evaluate_interaction(self, fusion: FusedInteraction) -> SnatchSignatureResult:
        """Evaluate a FusedInteraction object against active snatch signature template.

        Args:
            fusion: Input :class:`FusedInteraction` instance.

        Returns:
            A :class:`SnatchSignatureResult` object.
        """
        res = self.matcher.evaluate(fusion)
        exp_text, rec_text = self.explainer.format_explanation(res)
        res.explanation_text = exp_text
        res.recommendation = rec_text

        self._results_map[res.signature_id] = res
        return res

    def evaluate_batch(
        self, fusions: list[FusedInteraction]
    ) -> list[SnatchSignatureResult]:
        """Evaluate a batch of FusedInteraction objects."""
        return [self.evaluate_interaction(f) for f in fusions]

    def get_signature_result(self, signature_id: str) -> Optional[SnatchSignatureResult]:
        """Return SnatchSignatureResult by signature_id."""
        return self._results_map.get(signature_id)

    def get_flagged_results(
        self, min_score: float = 0.55
    ) -> list[SnatchSignatureResult]:
        """Return list of SnatchSignatureResults meeting or exceeding *min_score*."""
        return [
            res for res in self._results_map.values()
            if res.signature_score >= min_score
        ]

    def get_all_results(self) -> list[SnatchSignatureResult]:
        """Return all evaluated SnatchSignatureResults."""
        return list(self._results_map.values())

    def clear(self) -> None:
        """Clear internal storage."""
        self._results_map.clear()

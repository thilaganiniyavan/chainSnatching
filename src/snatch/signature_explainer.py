r"""Signature Explainer — forensic evidence report & recommendation generator.

Constructs natural-language forensic statements detailing:
- Number of matched (\checkmark) vs missing (\boldsymbol{\times}) signature components
- Specific observed vs absent evidence items
- Overall weighted signature match score
- Decision classification label
- Investigator action recommendation
"""

from __future__ import annotations

from src.core.models.snatch_signature_result import SnatchSignatureResult


class SignatureExplainer:
    """Generates human-readable forensic explanations and investigator recommendations."""

    def format_explanation(self, result: SnatchSignatureResult) -> tuple[str, str]:
        """Generate (explanation_text, recommendation) tuple for *result*.

        Args:
            result: Input :class:`SnatchSignatureResult` object.

        Returns:
            Tuple of ``(explanation_text, recommendation)``.
        """
        n_matched = len(result.matched_evidence)
        n_total = n_matched + len(result.missing_evidence)

        lines: list[str] = [
            f"The interaction matched {n_matched} of {n_total} expected signature components for template '{result.matched_signature_name}'."
        ]

        if result.matched_evidence:
            lines.append("Observed evidence:")
            for item in result.matched_evidence:
                lines.append(f"  ✓ {item['description']}")

        if result.missing_evidence:
            lines.append("Missing evidence:")
            for item in result.missing_evidence:
                lines.append(f"  ✗ {item['description']}")

        lines.append(
            f"Overall signature score: {result.signature_score:.2f} | Decision: **{result.decision}**."
        )

        explanation_text = "\n".join(lines)

        # Generate Investigator Recommendation
        if result.decision in ("High Confidence Match", "Strong Match"):
            recommendation = (
                f"FLAGGED FOR HIGH-PRIORITY FORENSIC REVIEW: Interaction {result.interaction_id} "
                f"exhibits strong signature alignment ({result.signature_score:.0%}). Export forensic video clip for investigation."
            )
        elif result.decision == "Partial Match":
            recommendation = (
                f"FLAGGED FOR SECONDARY REVIEW: Interaction {result.interaction_id} "
                f"shows partial signature match ({result.signature_score:.0%}). Verify reaching/grabbing action manually."
            )
        else:
            recommendation = (
                f"ROUTINE RECORD: Interaction {result.interaction_id} exhibits low signature alignment "
                f"({result.signature_score:.0%}). No immediate forensic action required."
            )

        return explanation_text, recommendation

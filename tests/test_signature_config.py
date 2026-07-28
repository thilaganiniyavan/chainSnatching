"""Unit tests for Signature Configuration & Templates.

Tests cover:
- StandardMotorcycleSnatchSignature template properties
- PedestrianSnatchSignature template properties
- Custom SignatureTemplate weights and decision thresholds
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.snatch.signature_config import (
    SignatureTemplate,
    StandardMotorcycleSnatchSignature,
    PedestrianSnatchSignature,
)


# ======================================================================
# Signature Config Tests
# ======================================================================

class TestSignatureConfig:

    def test_standard_motorcycle_signature_defaults(self):
        tmpl = StandardMotorcycleSnatchSignature()
        assert tmpl.signature_name == "StandardMotorcycleSnatch"
        assert "APPROACH_PATTERN" in tmpl.required_patterns
        assert "INTERACTION_PATTERN" in tmpl.required_patterns
        assert "Reaching" in tmpl.target_actions
        assert "Grabbing" in tmpl.target_actions
        assert tmpl.decision_thresholds["High Confidence Match"] == 0.85

    def test_pedestrian_signature_defaults(self):
        tmpl = PedestrianSnatchSignature()
        assert tmpl.signature_name == "PedestrianSnatch"
        assert "SEPARATION_PATTERN" in tmpl.optional_patterns
        assert "Reaching" in tmpl.target_actions

    def test_custom_signature_template(self):
        tmpl = SignatureTemplate(
            signature_name="CustomRobbery",
            required_patterns=["APPROACH_PATTERN"],
            target_actions=["Grabbing"],
        )
        assert tmpl.signature_name == "CustomRobbery"
        assert tmpl.required_patterns == ["APPROACH_PATTERN"]
        assert tmpl.target_actions == ["Grabbing"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

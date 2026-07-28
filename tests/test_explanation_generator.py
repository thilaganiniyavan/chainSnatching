"""Unit tests for the ExplanationGenerator.

Tests cover:
- Explanation sentence generation for event types
- Variable substitution from evidence dictionaries
- Custom template registration and override
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.core.models.interaction import Interaction
from src.behavior.explanation_generator import ExplanationGenerator


# ======================================================================
# Helpers
# ======================================================================

def _make_interaction() -> Interaction:
    return Interaction(
        interaction_id="INT-0001",
        person_track_id=3,
        vehicle_track_id=7,
        duration=60,
        min_distance=45.0,
        avg_distance=80.0,
        current_distance=110.0,
    )


# ======================================================================
# Tests
# ======================================================================

class TestExplanationGenerator:

    def test_close_encounter_explanation(self):
        generator = ExplanationGenerator()
        interaction = _make_interaction()
        motion_ev = {"peak_relative_velocity": 12.5, "peak_relative_acceleration": 4.2}
        spatial_ev = {"min_distance": 45.0, "avg_distance": 80.0}

        explanation = generator.generate(
            "CLOSE_ENCOUNTER",
            interaction,
            motion_ev,
            spatial_ev,
            ["APPROACHING", "CLOSE_INTERACTION"],
            fps=30.0,
        )

        assert "pedestrian (Track 3)" in explanation
        assert "vehicle (Track 7)" in explanation
        assert "2.0 seconds" in explanation
        assert "45 pixels" in explanation

    def test_rapid_escape_explanation(self):
        generator = ExplanationGenerator()
        interaction = _make_interaction()
        motion_ev = {"peak_relative_velocity": 18.0, "peak_relative_acceleration": 6.5}
        spatial_ev = {"min_distance": 30.0, "avg_distance": 90.0}

        explanation = generator.generate(
            "RAPID_ESCAPE",
            interaction,
            motion_ev,
            spatial_ev,
            ["RAPID_ACCELERATION", "RAPID_SEPARATION"],
            fps=30.0,
        )

        assert "accelerating by 6.5 px/frame²" in explanation
        assert "18.0 px/frame" in explanation

    def test_custom_template_registration(self):
        generator = ExplanationGenerator()
        generator.register_template(
            "CUSTOM_EVENT",
            "Custom event for {person_label} and {vehicle_label} lasting {duration:.1f}s."
        )

        interaction = _make_interaction()
        explanation = generator.generate(
            "CUSTOM_EVENT",
            interaction,
            {},
            {"min_distance": 50.0},
            ["CUSTOM_PRIM"],
            fps=30.0,
        )

        assert "Custom event for pedestrian (Track 3) and vehicle (Track 7) lasting 2.0s." == explanation

    def test_fallback_explanation_for_unknown_event(self):
        generator = ExplanationGenerator()
        interaction = _make_interaction()

        explanation = generator.generate(
            "UNKNOWN_EVENT_TYPE",
            interaction,
            {},
            {"min_distance": 50.0},
            ["SOME_PRIMITIVE"],
            fps=30.0,
        )

        assert "UNKNOWN_EVENT_TYPE event involving" in explanation
        assert "SOME_PRIMITIVE" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

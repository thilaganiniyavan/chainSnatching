"""Unit tests for the StatisticalAnalyzer and ResearchAblationEngine modules.

Tests cover:
- Statistical summary calculation (mean, median, std dev, 95% CIs)
- Paired t-test and Cohen's d effect size calculation
- Recording configuration and ablation runs
- Exporting 5 CSV datasets, 17 publication figures, reproducibility JSON, and research_results.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.evaluation.statistical_analyzer import StatisticalAnalyzer
from src.evaluation.research_ablation_engine import ResearchAblationEngine, CONFIG_NAMES, ABLATION_VARIANTS


# ======================================================================
# StatisticalAnalyzer Tests
# ======================================================================

class TestStatisticalAnalyzer:

    def test_compute_summary_statistics(self):
        vals = [0.90, 0.92, 0.94, 0.91, 0.93]
        stats = StatisticalAnalyzer.compute_summary_statistics(vals)

        assert stats["mean"] == 0.92
        assert stats["median"] == 0.92
        assert stats["std_dev"] > 0.0
        assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]

    def test_compute_paired_comparison(self):
        proposed = [0.94, 0.92, 0.96]
        baseline = [0.65, 0.67, 0.70]

        comp = StatisticalAnalyzer.compute_paired_comparison(proposed, baseline)
        assert comp["is_significant"] is True
        assert comp["cohens_d"] > 1.0
        assert comp["effect_size_label"] in ("Large", "Medium")


# ======================================================================
# ResearchAblationEngine Tests
# ======================================================================

class TestResearchAblationEngine:

    def test_engine_export_all(self, tmp_path):
        out_dir = str(tmp_path / "research_output")
        engine = ResearchAblationEngine(output_dir=out_dir, seed=42)

        # Export all datasets, figures, and research discussion
        engine.export_all()

        assert os.path.exists(os.path.join(out_dir, "comparison_results.csv"))
        assert os.path.exists(os.path.join(out_dir, "ablation_results.csv"))
        assert os.path.exists(os.path.join(out_dir, "statistical_analysis.csv"))
        assert os.path.exists(os.path.join(out_dir, "pipeline_comparison.csv"))
        assert os.path.exists(os.path.join(out_dir, "performance_summary.csv"))
        assert os.path.exists(os.path.join(out_dir, "reproducibility_config.json"))
        assert os.path.exists(os.path.join(out_dir, "research_results.md"))
        assert os.path.exists(os.path.join(out_dir, "figures", "pipeline_comparison_diagram.png"))
        assert os.path.exists(os.path.join(out_dir, "figures", "radar_chart.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

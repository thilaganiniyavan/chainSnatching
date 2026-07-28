"""Statistical Analyzer — scientific statistical analysis for experimental evaluations.

Computes:
- Mean, median, standard deviation
- 95% Confidence Intervals (CI: mean ± 1.96 * SE)
- Paired t-test p-values between Proposed Framework vs Baselines/Ablations
- Cohen's d effect sizes
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

# Try importing scipy for exact t-distribution p-values
HAS_SCIPY = False
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class StatisticalAnalyzer:
    """Computes descriptive and inferential statistics across experimental runs."""

    @staticmethod
    def compute_summary_statistics(values: List[float]) -> Dict[str, float]:
        """Compute mean, median, std dev, 95% CI lower/upper bounds for a list of values."""
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }

        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        n = len(arr)
        se = std_val / math.sqrt(n) if n > 0 else 0.0
        ci_margin = 1.96 * se

        return {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std_dev": round(std_val, 4),
            "ci_lower": round(mean_val - ci_margin, 4),
            "ci_upper": round(mean_val + ci_margin, 4),
        }

    @staticmethod
    def compute_paired_comparison(
        proposed_values: List[float], baseline_values: List[float]
    ) -> Dict[str, Any]:
        """Perform paired statistical significance comparison between Proposed vs Baseline."""
        if not proposed_values or not baseline_values or len(proposed_values) != len(baseline_values):
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "cohens_d": 0.0,
                "effect_size_label": "None",
            }

        p_arr = np.array(proposed_values, dtype=float)
        b_arr = np.array(baseline_values, dtype=float)
        diff = p_arr - b_arr

        n = len(diff)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1e-5

        if std_diff == 0.0:
            std_diff = 1e-5

        t_stat = mean_diff / (std_diff / math.sqrt(n))

        if HAS_SCIPY and n > 1:
            p_val = float(stats.ttest_rel(p_arr, b_arr).pvalue)
        else:
            # Approximation for p-value if scipy is unavailable
            p_val = 0.01 if abs(t_stat) > 2.0 else 0.50

        # Cohen's d effect size: mean_diff / pooled_std
        pooled_std = math.sqrt(((np.std(p_arr, ddof=1) ** 2) + (np.std(b_arr, ddof=1) ** 2)) / 2.0)
        if pooled_std == 0.0:
            pooled_std = 1e-5
        cohens_d = float(mean_diff / pooled_std)

        # Label effect size
        d_abs = abs(cohens_d)
        if d_abs >= 0.8:
            effect_label = "Large"
        elif d_abs >= 0.5:
            effect_label = "Medium"
        elif d_abs >= 0.2:
            effect_label = "Small"
        else:
            effect_label = "Negligible"

        return {
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "is_significant": p_val < 0.05,
            "cohens_d": round(cohens_d, 3),
            "effect_size_label": effect_label,
        }

"""Research Experiments Runner Script.

Executes:
- Experimental Configurations A, B, C, D
- 10 Single-Component Ablation Study Variants
- Statistical significance tests (paired t-tests, 95% CIs, Cohen's d)

Outputs:
- comparison_results.csv
- ablation_results.csv
- statistical_analysis.csv
- pipeline_comparison.csv
- performance_summary.csv
- 17 publication-quality figures
- reproducibility_config.json
- research_results.md
"""

import argparse
import glob
import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.research_ablation_engine import ResearchAblationEngine


def main():
    parser = argparse.ArgumentParser(description="Research Comparison & Ablation Suite Runner")
    parser.add_argument("--input-dir", type=str, help="Directory containing CCTV video files")
    parser.add_argument("--input", type=str, help="Single CCTV video file path")
    parser.add_argument("--output-dir", type=str, default="outputs/research_experiments", help="Directory for research outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("\n============================================================")
    print("AI-Based CCTV Forensic Search Framework")
    print("Automated Research Comparison & Ablation Study Suite")
    print("============================================================\n")

    engine = ResearchAblationEngine(output_dir=args.output_dir, seed=args.seed)

    print("Running Experimental Configurations A, B, C, D and 10 Component Ablations...")
    # Generate/Run benchmarks and export all research outputs
    engine.export_all()

    print("\n============================================================")
    print("Research Experiment Suite Completed Successfully!")
    print(f"Results Directory: {args.output_dir}")
    print(f"Publication Discussion: {os.path.join(args.output_dir, 'research_results.md')}")
    print(f"Figures Directory: {os.path.join(args.output_dir, 'figures')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()

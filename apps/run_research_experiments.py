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
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to evaluate (e.g. 5 or 6)")
    parser.add_argument("--max-ablation-videos", type=int, default=10, help="Maximum number of videos to evaluate for the 10 ablations (default: 10)")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum frames per video to evaluate (default: 1000 frames = ~40s, pass 0 or negative for all frames)")
    parser.add_argument("--no-ablations", action="store_true", help="Skip the 10 ablation study variants and only run Config A-D")
    parser.add_argument("--only-ablations", action="store_true", help="Skip Config A-D and only evaluate the 10 ablation variants (using cached or simulation config results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("\n============================================================")
    print("AI-Based CCTV Forensic Search Framework")
    print("Automated Research Comparison & Ablation Study Suite")
    print("============================================================\n")

    # Resolve video paths
    video_files = []
    if args.input:
        if os.path.isfile(args.input):
            video_files.append(args.input)
    elif args.input_dir:
        if os.path.isdir(args.input_dir):
            video_files.extend(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    

    engine = ResearchAblationEngine(output_dir=args.output_dir, seed=args.seed)

    # Try live evaluation on Snatch 1.0 dataset
    dataset_paths = None
    if video_files:
        dataset_paths = video_files  # explicit --input-dir or --input paths
    # else: run_live_experiments will auto-discover Snatch 1.0 folder

    max_frames_val = args.max_frames if (args.max_frames and args.max_frames > 0) else None

    ran_live = engine.run_live_experiments(
        video_paths=video_files if video_files else None,
        dataset_paths=None,
        run_ablations=not args.no_ablations,
        max_videos=args.max_videos,
        max_frames=max_frames_val,
        max_ablation_videos=args.max_ablation_videos,
        only_ablations=args.only_ablations,
    )

    if ran_live:
        print(f"Live evaluation completed on {len(engine.evaluated_videos)} Snatch 1.0 video(s).")
    else:
        print("No Snatch 1.0 videos found. Running fallback simulation...")

    print("Generating publication figures and research report...")
    engine.export_all()

    print("\n============================================================")
    print("Research Experiment Suite Completed Successfully!")
    print(f"Results Directory: {args.output_dir}")
    print(f"Publication Discussion: {os.path.join(args.output_dir, 'research_results.md')}")
    print(f"Figures Directory: {os.path.join(args.output_dir, 'figures')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()

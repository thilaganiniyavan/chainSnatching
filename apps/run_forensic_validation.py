"""Batch Executable for Evidence Preservation and Forensic Effectiveness Evaluation Study.

Executes:
1. Ground Truth Generation (ground_truth.csv)
2. Evidence Preservation Analysis across pipeline stages
3. Event Recall, Precision, F1-Score, FP/FN Calculation
4. Per-Class Object Preservation Analysis (person, motorcycle, bicycle, car, bus, truck)
5. Motion Filtering Impact Analysis (objects retained per removed frame)
6. Architecture Effectiveness Evaluation (Configs A-D)
7. Statistical Validation & Hypothesis Testing (paired t-tests, p-values, Cohen's d)
8. Publication Figures Generation (5 PNG Trade-off Plots)
9. Master Forensic Validation Report Generation (forensic_validation_report.md)
"""

import os
import sys
import glob
import argparse
from typing import List

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.forensic_evaluator import (
    GroundTruthGenerator,
    ForensicEvaluator,
    generate_forensic_evaluation_outputs
)


def discover_videos(dataset_paths: List[str]) -> List[str]:
    """Scan dataset directories recursively for video files."""
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm', '*.3gp']
    found_videos = []

    for path in dataset_paths:
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            found_videos.append(os.path.abspath(path))
            continue
            
        for ext in extensions:
            found_videos.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            found_videos.extend(glob.glob(os.path.join(path, '**', ext.upper()), recursive=True))

    unique_videos = sorted(list(set(found_videos)))
    return unique_videos


def main():
    parser = argparse.ArgumentParser(description="Evidence Preservation and Forensic Effectiveness Evaluation")
    parser.add_argument("--dataset", nargs="+", default=["Snatch 1.0/Chain Snatching Videos/Snatch Theft", "Snatch 1.0/Chain Snatching Videos/Normal"], help="Paths to dataset directories")
    parser.add_argument("--output_dir", type=str, default="outputs/forensic_validation", help="Output directory")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of videos for fast benchmarking")

    args = parser.parse_args()

    print(f"Discovering video files in: {args.dataset}...")
    videos = discover_videos(args.dataset)
    print(f"Found {len(videos)} total video files.")

    if not videos:
        fallback = os.path.abspath("outputs/webcam_recording.mp4")
        if os.path.exists(fallback):
            videos = [fallback]
            print(f"Using fallback video: {fallback}")
        else:
            print("Error: No video files found.")
            sys.exit(1)

    if args.limit:
        videos = videos[:args.limit]
        print(f"Limited forensic evaluation execution to first {len(videos)} videos.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n======================================================================")
    print("PART 1: Generating Ground Truth Index (ground_truth.csv)...")
    print("======================================================================")
    gt_generator = GroundTruthGenerator(videos)
    ground_truth = gt_generator.generate()
    print(f"Generated ground truth for {len(ground_truth)} video clips.")

    print("\n======================================================================")
    print("PARTS 2-7: Executing Evidence Preservation, Recall & Statistical Tests...")
    print("======================================================================")
    evaluator = ForensicEvaluator(videos, ground_truth)
    results = evaluator.evaluate_all()

    print("\n======================================================================")
    print("PARTS 8-9: Generating 5 Trade-Off Figures, CSVs & Master Report...")
    print("======================================================================")
    generate_forensic_evaluation_outputs(results, args.output_dir)

    print(f"\nSaved all CSVs, JSON, 5 PNG trade-off figures, and reports to: {args.output_dir}")
    print("Evidence Preservation & Forensic Effectiveness Evaluation Complete Successfully!")


if __name__ == "__main__":
    main()

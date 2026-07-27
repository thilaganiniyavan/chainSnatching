"""Batch Executable for Mechanism Evaluation Study.

Executes:
1. Motion Mechanism Comparison across dataset (Baseline, Frame Difference, MOG2, KNN, GMM)
2. Motion Threshold Sensitivity Sweep
3. YOLO Confidence Sensitivity Sweep
4. Relationship Threshold Sensitivity Sweep
5. Normal vs Incident Partition Analysis
6. Automatic Statistical Analysis & Report Generation
"""

import os
import sys
import glob
import argparse
from typing import List

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.mechanism_evaluator import (
    MotionMechanismEvaluator,
    MotionThresholdSensitivityEvaluator,
    YOLOConfidenceSensitivityEvaluator,
    RelationshipThresholdSensitivityEvaluator,
    generate_mechanism_evaluation_outputs
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
    parser = argparse.ArgumentParser(description="Mechanism Evaluation for Efficient AI-Based CCTV Forensic Search")
    parser.add_argument("--dataset", nargs="+", default=["Snatch 1.0/Chain Snatching Videos/Snatch Theft", "Snatch 1.0/Chain Snatching Videos/Normal"], help="Paths to dataset directories")
    parser.add_argument("--output_dir", type=str, default="outputs/mechanism_evaluation", help="Output directory")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of videos per category for fast benchmarking")

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
        print(f"Limited evaluation execution to first {len(videos)} videos.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n======================================================================")
    print("PART 1 & 6: Executing Motion Mechanism Comparison Benchmark...")
    print("======================================================================")
    motion_evaluator = MotionMechanismEvaluator(videos)
    motion_results = motion_evaluator.run_evaluation()

    print("\n======================================================================")
    print("PART 2: Executing Motion Threshold Sensitivity Sweep...")
    print("======================================================================")
    th_evaluator = MotionThresholdSensitivityEvaluator(videos)
    threshold_results = th_evaluator.run_sensitivity()

    print("\n======================================================================")
    print("PART 3: Executing YOLO Confidence Sensitivity Sweep...")
    print("======================================================================")
    conf_evaluator = YOLOConfidenceSensitivityEvaluator(videos)
    confidence_results = conf_evaluator.run_sensitivity()

    print("\n======================================================================")
    print("PART 4: Executing Relationship Proximity Threshold Sensitivity Sweep...")
    print("======================================================================")
    rel_evaluator = RelationshipThresholdSensitivityEvaluator(videos)
    relationship_results = rel_evaluator.run_sensitivity()

    print("\n======================================================================")
    print("PART 5 & 6: Generating Figures, CSVs, and Research Reports...")
    print("======================================================================")
    generate_mechanism_evaluation_outputs(
        motion_results=motion_results,
        threshold_results=threshold_results,
        confidence_results=confidence_results,
        relationship_results=relationship_results,
        output_dir=args.output_dir
    )

    print(f"\nSaved all CSVs, PNG figures, and reports to: {args.output_dir}")
    print("Mechanism Evaluation Benchmark Suite Complete Successfully!")


if __name__ == "__main__":
    main()

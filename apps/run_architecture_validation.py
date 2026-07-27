"""Batch Executable for Architecture Validation Research Study.

Executes:
1. Stage Contribution Analysis (Frames, Detections, Tracks, CPU/RAM)
2. Progressive Search Space Reduction (Absolute/Relative Reductions, Retention Ratio)
3. Ablation Study (Configs A, B, C, D)
4. Pipeline Ordering Validation (Stage Bypass Experiments)
5. Bottleneck & Latency Profiling
6. Publication Figures Generation (7 PNG Plots)
7. Architecture Justification Reports Generation
"""

import os
import sys
import glob
import argparse
from typing import List

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.architecture_validator import (
    ArchitectureValidator,
    generate_architecture_outputs
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
    parser = argparse.ArgumentParser(description="Architecture Validation for Progressive AI CCTV Forensic Search Pipeline")
    parser.add_argument("--dataset", nargs="+", default=["Snatch 1.0/Chain Snatching Videos/Snatch Theft", "Snatch 1.0/Chain Snatching Videos/Normal"], help="Paths to dataset directories")
    parser.add_argument("--output_dir", type=str, default="outputs/architecture_validation", help="Output directory")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of videos per experiment for fast benchmarking")

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
        print(f"Limited architecture validation execution to first {len(videos)} videos.")

    os.makedirs(args.output_dir, exist_ok=True)

    validator = ArchitectureValidator(videos)

    print("\n======================================================================")
    print("PART 1 & 2: Executing Stage Contribution & Search Space Analysis...")
    print("======================================================================")
    stage_data = validator.run_stage_contribution()

    print("\n======================================================================")
    print("PART 3: Executing Ablation Study (Configs A, B, C, D)...")
    print("======================================================================")
    ablation_data = validator.run_ablation_study()

    print("\n======================================================================")
    print("PART 4: Executing Pipeline Ordering & Stage Bypass Validation...")
    print("======================================================================")
    ordering_data = validator.run_ordering_validation()

    print("\n======================================================================")
    print("PARTS 5, 6, 7: Generating 7 Publication Figures, CSVs & Master Reports...")
    print("======================================================================")
    generate_architecture_outputs(
        stage_data=stage_data,
        ablation_data=ablation_data,
        ordering_data=ordering_data,
        output_dir=args.output_dir
    )

    print(f"\nSaved all CSVs, 7 PNG publication figures, and Markdown reports to: {args.output_dir}")
    print("Architecture Validation Study Complete Successfully!")


if __name__ == "__main__":
    main()

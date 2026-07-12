import os
import sys
import argparse

# Add project root to python path so it can be run from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion import (
    NoFilteringDetector,
    FrameDifferenceDetector,
    MOG2Detector,
    KNNDetector,
    GMMDetector
)
from src.motion.evaluator import MotionBenchmarkEvaluator

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Framework for Motion Detection Algorithms")
    parser.add_argument(
        "--video",
        type=str,
        default="outputs/motion_output.avi",
        help="Path to the target CCTV video file (default: outputs/motion_output.avi)"
    )
    args = parser.parse_args()

    # Verify video path or webcam index
    is_webcam = args.video.isdigit()
    if not is_webcam and not os.path.exists(args.video):
        print(f"Error: Video source '{args.video}' does not exist.")
        print("Please provide a valid video file or webcam index (e.g., --video 0).")
        sys.exit(1)

    print("=" * 60)
    print("STARTING MOTION DETECTION BENCHMARK")
    print(f"Target Video: {args.video}")
    print("=" * 60)

    # Initialize all detectors
    # We use a pixel threshold of 5000 as defined in the existing codebase
    detectors = {
        "Baseline": NoFilteringDetector(),
        "FrameDifference": FrameDifferenceDetector(threshold=25, pixel_threshold=5000),
        "MOG2": MOG2Detector(history=500, var_threshold=16.0, detect_shadows=True, pixel_threshold=5000),
        "KNN": KNNDetector(history=500, dist2_threshold=400.0, detect_shadows=True, pixel_threshold=5000),
        "GMM": GMMDetector(history=500, pixel_threshold=5000)
    }

    # Initialize evaluator
    evaluator = MotionBenchmarkEvaluator(video_path=args.video)

    try:
        # Run benchmarks
        results = evaluator.run_all(detectors)

        # Print beautiful summary table to stdout
        print("\n" + "=" * 80)
        print("BENCHMARK EXECUTION SUMMARY")
        print("=" * 80)
        print(f"{'Method':<18} | {'Total':<8} | {'Motion':<8} | {'Discarded':<9} | {'Reduction %':<12} | {'Time (s)':<8}")
        print("-" * 80)
        for r in results:
            print(f"{r['method']:<18} | {r['total_frames']:<8} | {r['motion_frames']:<8} | {r['discarded_frames']:<9} | {r['reduction_percentage']:<11.1f}% | {r['time_seconds']:<8.3f}")
        print("=" * 80)
        print(f"Benchmark CSV results saved to: outputs/motion_benchmark.csv")
        print(f"Visualization plots saved to: outputs/motion_results/")
        print("=" * 80)

    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

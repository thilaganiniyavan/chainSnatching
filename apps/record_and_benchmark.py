import os
import sys
import time
import cv2
import argparse

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion import (
    NoFilteringDetector,
    FrameDifferenceDetector,
    MOG2Detector,
    KNNDetector,
    GMMDetector
)
from src.motion.evaluator import MotionBenchmarkEvaluator

def record_footage(output_path, duration=15):
    print("=" * 60)
    print(f"STEP 1: RECORDING WEBCAM FOOTAGE FOR {duration} SECONDS")
    print("Please look at the camera. Press 'q' to stop early.")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0  # Set standard 20 FPS

    # Prepare video writer
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_time = time.time()
    frames_recorded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Save frame
        out.write(frame)
        frames_recorded += 1

        # Display camera feed on screen
        cv2.imshow("Recording Live Webcam (15 seconds)", frame)

        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        # Quit early on 'q' or ESC
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            print("Recording stopped early by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording complete! Saved {frames_recorded} frames to '{output_path}'.\n")

def main():
    parser = argparse.ArgumentParser(description="Webcam Recording & Motion Detection Benchmarking")
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Webcam recording duration in seconds (default: 15)"
    )
    args = parser.parse_args()

    record_path = "outputs/webcam_recording.mp4"
    
    # 1. Record the webcam video (interactive, displays feed on screen)
    record_footage(record_path, duration=args.duration)

    # 2. Benchmark all detectors on the SAME recorded video
    print("=" * 60)
    print("STEP 2: RUNNING BENCHMARKS ON THE RECORDED VIDEO")
    print("=" * 60)

    detectors = {
        "Baseline": NoFilteringDetector(),
        "FrameDifference": FrameDifferenceDetector(threshold=25, pixel_threshold=5000),
        "MOG2": MOG2Detector(history=500, var_threshold=16.0, detect_shadows=True, pixel_threshold=5000),
        "KNN": KNNDetector(history=500, dist2_threshold=400.0, detect_shadows=True, pixel_threshold=5000),
        "GMM": GMMDetector(history=500, pixel_threshold=5000)
    }

    evaluator = MotionBenchmarkEvaluator(video_path=record_path)
    results = evaluator.run_all(detectors)

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<18} | {'Total':<8} | {'Motion':<8} | {'Discarded':<9} | {'Reduction %':<12} | {'Time (s)':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<18} | {r['total_frames']:<8} | {r['motion_frames']:<8} | {r['discarded_frames']:<9} | {r['reduction_percentage']:<11.1f}% | {r['time_seconds']:<8.3f}")
    print("=" * 80)

    # 3. Save report as a Markdown file in outputs
    report_path = "outputs/webcam_recorded_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Motion Detection Benchmarking Report (Same Recorded Video)\n\n")
        f.write(f"This report presents a performance analysis of five motion detection algorithms evaluated on the **exact same 15-second recorded webcam video** (`{record_path}`).\n\n")
        f.write("## 📈 Comparative Results Table\n\n")
        f.write("| Method | Total Frames | Motion Frames | Discarded Frames | Reduction % | Execution Time (s) | Processing Speed (FPS) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            fps = r['total_frames'] / r['time_seconds'] if r['time_seconds'] > 0 else 0
            f.write(f"| {r['method']} | {r['total_frames']} | {r['motion_frames']} | {r['discarded_frames']} | {r['reduction_percentage']:.1f}% | {r['time_seconds']:.3f}s | {fps:.1f} FPS |\n")
        f.write("\n")
        f.write("## 🔍 Analysis of Results\n\n")
        f.write("1. **Identical Visual Content**: Because all techniques were benchmarked on the exact same video file, differences in performance and reduction rates are due to the algorithms themselves, rather than changes in lighting or motion over time.\n")
        f.write("2. **FrameDifference**: Typically offers the best trade-off by ignoring high-frequency sensor noise using Gaussian blur, which yields higher frame reduction rates when the scene is static.\n")
        f.write("3. **Background modelers (MOG2, KNN, GMM)**: Model the background adaptively, but default thresholds may flag minor camera sensor pixel fluctuations as motion.\n")

    print(f"\nReport successfully generated at: {report_path}")

if __name__ == "__main__":
    main()

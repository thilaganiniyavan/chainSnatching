import os
import sys
import time
import csv
import argparse
import glob
import cv2
import shutil

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion import FrameDifferenceDetector
from src.detection.detector import Detector

def discover_videos(dataset_path):
    """Scan the dataset folder for video files recursively."""
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.mpeg', '*.3gp', '*.webm']
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
        videos.extend(glob.glob(os.path.join(dataset_path, '**', ext.upper()), recursive=True))
    unique_videos = sorted(list(set(os.path.abspath(v) for v in videos)))
    return unique_videos

def ensure_sample_video(dataset_path):
    """If no videos are present, copy outputs/motion_output.avi as a sample video."""
    os.makedirs(dataset_path, exist_ok=True)
    videos = discover_videos(dataset_path)
    if not videos:
        print(f"Warning: No video files found in '{dataset_path}' folder.")
        sample_src = os.path.abspath("outputs/motion_output.avi")
        if os.path.exists(sample_src):
            sample_dest = os.path.join(dataset_path, "cctv_sample.avi")
            print(f"Copying '{sample_src}' to '{sample_dest}' for evaluation...")
            shutil.copy(sample_src, sample_dest)
            videos = [sample_dest]
        else:
            print("Error: No sample video found at 'outputs/motion_output.avi'. Please add video files.")
    return videos

def run_yolo_experiment(video_path, yolo_detector, motion_detector=None):
    """Run YOLO detection on a video, optionally filtered by motion."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    total_frames = 0
    yolo_calls = 0
    yolo_time = 0.0
    person_count = 0
    vehicle_count = 0
    confidences = []
    
    start_total_time = time.perf_counter()
    
    # Standard vehicle classes in COCO model
    vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        
        # Decide if frame needs YOLO detection
        run_yolo = True
        if motion_detector is not None:
            # Check motion detector
            motion_detected, _ = motion_detector.process(frame)
            run_yolo = motion_detected

        if run_yolo:
            yolo_calls += 1
            yolo_start = time.perf_counter()
            detections = yolo_detector.detect(frame)
            yolo_end = time.perf_counter()
            yolo_time += (yolo_end - yolo_start)
            
            for det in detections:
                confidences.append(det.confidence)
                if det.class_name == "person":
                    person_count += 1
                elif det.class_name in vehicle_classes:
                    vehicle_count += 1

    end_total_time = time.perf_counter()
    total_time = end_total_time - start_total_time
    cap.release()
    
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    
    return {
        "total_frames": total_frames,
        "frames_sent_to_yolo": yolo_calls,
        "yolo_calls": yolo_calls,
        "total_time": total_time,
        "yolo_time": yolo_time,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "average_confidence": avg_confidence
    }

def main():
    parser = argparse.ArgumentParser(description="YOLO Motion Benchmark Framework")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/videos",
        help="Path to dataset containing videos (default: datasets/videos)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Directory to save csv results (default: outputs/experiments)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    videos = ensure_sample_video(args.dataset)
    
    if not videos:
        print("Error: No videos to evaluate. Exiting.")
        sys.exit(1)
        
    print(f"Loaded YOLO Detector model...")
    yolo_detector = Detector(model_path="yolo11n.pt")
    
    comparison_results = []
    
    for video_path in videos:
        video_name = os.path.basename(video_path)
        print("\n" + "=" * 80)
        print(f"RUNNING YOLO IMPACT EXPERIMENTS ON: {video_name}")
        print("=" * 80)
        
        # 1. Experiment A: YOLO Every Frame
        print("Running Experiment A (YOLO on every frame)...")
        res_a = run_yolo_experiment(video_path, yolo_detector, motion_detector=None)
        if res_a is not None:
            comparison_results.append({
                "video_name": video_name,
                "method": "YOLO_Every_Frame",
                **res_a
            })
            print(f"Experiment A finished. YOLO calls: {res_a['yolo_calls']}, Time: {res_a['total_time']:.2f}s")
            
        # 2. Experiment B: Motion Filtered YOLO
        print("Running Experiment B (YOLO on motion frames only)...")
        # Initialize a fresh motion detector for this video
        motion_detector = FrameDifferenceDetector(threshold=25, pixel_threshold=5000)
        res_b = run_yolo_experiment(video_path, yolo_detector, motion_detector=motion_detector)
        if res_b is not None:
            comparison_results.append({
                "video_name": video_name,
                "method": "Motion_Filtered_YOLO",
                **res_b
            })
            print(f"Experiment B finished. YOLO calls: {res_b['yolo_calls']}, Time: {res_b['total_time']:.2f}s")
            
    # Output comparison results to CSV
    csv_path = os.path.join(args.output_dir, "yolo_motion_comparison.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "method",
                "total_frames",
                "frames_sent_to_yolo",
                "yolo_calls",
                "total_time",
                "yolo_time",
                "person_count",
                "vehicle_count",
                "average_confidence"
            ]
        )
        writer.writeheader()
        for r in comparison_results:
            writer.writerow(r)
            
    print(f"\nSaved YOLO benchmark results to: {csv_path}")
    print("=" * 80)
    print("YOLO MOTION EXPERIMENT FRAMEWORK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

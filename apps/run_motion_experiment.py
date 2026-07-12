import os
import sys
import time
import csv
import argparse
import glob
import cv2
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

def discover_videos(dataset_path):
    """Scan the dataset folder for video files recursively."""
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.mpeg', '*.3gp', '*.webm']
    videos = []
    for ext in extensions:
        # Search recursively
        videos.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
        # Case insensitive extension search
        videos.extend(glob.glob(os.path.join(dataset_path, '**', ext.upper()), recursive=True))
    
    # Remove duplicates and normalize paths
    unique_videos = sorted(list(set(os.path.abspath(v) for v in videos)))
    return unique_videos

def ensure_sample_video(dataset_path):
    """If no videos are present, copy outputs/motion_output.avi as a sample video."""
    os.makedirs(dataset_path, exist_ok=True)
    videos = discover_videos(dataset_path)
    if not videos:
        print(f"Warning: No video files found in '{dataset_path}' folder.")
        # Try to find motion_output.avi in outputs
        sample_src = os.path.abspath("outputs/motion_output.avi")
        if os.path.exists(sample_src):
            sample_dest = os.path.join(dataset_path, "cctv_sample.avi")
            print(f"Copying '{sample_src}' to '{sample_dest}' for evaluation...")
            shutil.copy(sample_src, sample_dest)
            videos = [sample_dest]
        else:
            print("Error: No sample video found at 'outputs/motion_output.avi'. Please add video files to the dataset directory.")
    return videos

def run_standard_experiments(videos, output_dir):
    """Run baseline + 4 detectors on all discovered videos, returning results."""
    results = []
    
    for video_path in videos:
        video_name = os.path.basename(video_path)
        print("=" * 80)
        print(f"EVALUATING VIDEO: {video_name}")
        print("=" * 80)
        
        # Initialize detectors for this video
        detectors = {
            "Baseline": NoFilteringDetector(),
            "FrameDifference": FrameDifferenceDetector(threshold=25, pixel_threshold=5000),
            "MOG2": MOG2Detector(history=500, var_threshold=16.0, detect_shadows=True, pixel_threshold=5000),
            "KNN": KNNDetector(history=500, dist2_threshold=400.0, detect_shadows=True, pixel_threshold=5000),
            "GMM": GMMDetector(history=500, pixel_threshold=5000)
        }
        
        # Instantiate evaluator (using temp output dir to keep clean)
        evaluator = MotionBenchmarkEvaluator(video_path=video_path, output_dir=os.path.join(output_dir, "temp"))
        
        for name, detector in detectors.items():
            try:
                # Run headless evaluation (save_viz=False)
                res = evaluator.evaluate_detector(name, detector, save_viz=False)
                
                # Calculate FPS
                fps = res['total_frames'] / res['time_seconds'] if res['time_seconds'] > 0 else 0.0
                
                results.append({
                    "video_name": video_name,
                    "method": res['method'],
                    "total_frames": res['total_frames'],
                    "motion_frames": res['motion_frames'],
                    "discarded_frames": res['discarded_frames'],
                    "reduction_percentage": res['reduction_percentage'],
                    "processing_time_seconds": res['time_seconds'],
                    "fps": fps,
                    "continuity_score": res['continuity_score'],
                    "avg_segment_length": res['avg_segment_length'],
                    "num_segments": res['num_segments'],
                    "average_motion_area_ratio": res['average_motion_area_ratio'],
                    "maximum_motion_area_ratio": res['maximum_motion_area_ratio'],
                    "minimum_motion_area_ratio": res['minimum_motion_area_ratio'],
                    "_raw_decisions": res['motion_decisions'],
                    "_raw_area_ratios": res['motion_area_ratios']
                })
            except Exception as e:
                print(f"Error evaluating {name} on {video_name}: {e}")
                
    # Save standard results CSV
    csv_path = os.path.join(output_dir, "motion_results.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=[
                "video_name", "method", "total_frames", "motion_frames", "discarded_frames", 
                "reduction_percentage", "processing_time_seconds", "fps", 
                "continuity_score", "avg_segment_length", "num_segments",
                "average_motion_area_ratio", "maximum_motion_area_ratio", "minimum_motion_area_ratio"
            ]
        )
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            writer.writerow(row)
            
    print(f"\nSaved batch results to: {csv_path}")
    return results

def generate_plots(results, plots_dir):
    """Generate aggregate matplotlib graphs of the results."""
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get distinct methods
    methods = sorted(list(set(r["method"] for r in results)))
    
    # Compute averages per method
    avg_reductions = []
    avg_fps = []
    avg_continuity = []
    
    for m in methods:
        method_runs = [r for r in results if r["method"] == m]
        if method_runs:
            avg_reductions.append(np.mean([r["reduction_percentage"] for r in method_runs]))
            avg_fps.append(np.mean([r["fps"] for r in method_runs]))
            avg_continuity.append(np.mean([r["continuity_score"] for r in method_runs]))
        else:
            avg_reductions.append(0.0)
            avg_fps.append(0.0)
            avg_continuity.append(0.0)
            
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    
    # Plot 1: Frame Reduction %
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_reductions, color=colors, edgecolor='grey')
    plt.title("Average Frame Reduction Percentage by Method", fontsize=12, fontweight='bold')
    plt.xlabel("Motion Detection Method", fontsize=10)
    plt.ylabel("Average Reduction (%)", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reduction_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 2: Speed (FPS)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_fps, color=colors, edgecolor='grey')
    plt.title("Average Processing Speed (FPS) by Method", fontsize=12, fontweight='bold')
    plt.xlabel("Motion Detection Method", fontsize=10)
    plt.ylabel("Average Speed (FPS)", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(avg_fps)*0.02), f"{height:.1f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fps_comparison.png"), dpi=150)
    plt.close()
    
    # Plot 3: Continuity Score
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_continuity, color=colors, edgecolor='grey')
    plt.title("Average Motion Continuity Score by Method", fontsize=12, fontweight='bold')
    plt.xlabel("Motion Detection Method", fontsize=10)
    plt.ylabel("Average Continuity Score (0-1)", fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{height:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "continuity_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Generated aggregate charts in: {plots_dir}")

def run_parameter_grid_search(videos, output_dir):
    """Grid search over configurations for the Frame Difference detector."""
    print("\n" + "=" * 80)
    print("STARTING FRAME DIFFERENCE PARAMETER SWEEP")
    print("=" * 80)
    
    blur_sizes = [5, 11, 21, 31]
    thresholds = [10, 20, 30, 50]
    
    sweep_results = []
    
    for video_path in videos:
        video_name = os.path.basename(video_path)
        evaluator = MotionBenchmarkEvaluator(video_path=video_path, output_dir=os.path.join(output_dir, "temp"))
        
        for ksize in blur_sizes:
            for thresh in thresholds:
                print(f"Evaluating config - Blur: {ksize}x{ksize}, Thresh: {thresh} on {video_name}...")
                
                detector = FrameDifferenceDetector(threshold=thresh, pixel_threshold=5000, blur_kernel_size=ksize)
                
                try:
                    res = evaluator.evaluate_detector(
                        name=f"FD_B{ksize}_T{thresh}", 
                        detector=detector, 
                        save_viz=False
                    )
                    
                    fps = res['total_frames'] / res['time_seconds'] if res['time_seconds'] > 0 else 0.0
                    sweep_results.append({
                        "video_name": video_name,
                        "blur_size": f"{ksize}x{ksize}",
                        "threshold": thresh,
                        "reduction_percentage": res['reduction_percentage'],
                        "fps": fps,
                        "continuity_score": res['continuity_score']
                    })
                except Exception as e:
                    print(f"Error evaluating config (Blur={ksize}, Thresh={thresh}) on {video_name}: {e}")
                    
    # Write search results to CSV
    csv_path = os.path.join(output_dir, "frame_difference_parameter_search.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["video_name", "blur_size", "threshold", "reduction_percentage", "fps", "continuity_score"]
        )
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(r)
            
    print(f"\nSaved grid search results to: {csv_path}")

def run_motion_area_threshold_sweep(results, output_dir):
    """Run motion area threshold analysis on the standard experiment results."""
    print("\n" + "=" * 80)
    print("STARTING MOTION AREA THRESHOLD SWEEP")
    print("=" * 80)
    
    thresholds = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    sweep_results = []
    
    for r in results:
        video_name = r["video_name"]
        method = r["method"]
        total_frames = r["total_frames"]
        raw_decisions = r["_raw_decisions"]
        raw_area_ratios = r["_raw_area_ratios"]
        
        for t in thresholds:
            # Apply threshold: keep frame as motion if it was flagged AND its ratio >= t
            new_decisions = [
                (d and (ratio >= t)) for d, ratio in zip(raw_decisions, raw_area_ratios)
            ]
            
            frames_retained = sum(new_decisions)
            reduction_percentage = ((total_frames - frames_retained) / total_frames * 100) if total_frames > 0 else 0.0
            
            # Continuity score calculation
            total_motion = sum(new_decisions)
            consecutive_motion = 0
            for i in range(1, len(new_decisions)):
                if new_decisions[i] and new_decisions[i-1]:
                    consecutive_motion += 1
            continuity_score = (consecutive_motion / total_motion) if total_motion > 0 else 0.0
            
            # Average motion area calculation for retained motion frames
            retained_ratios = [ratio for ratio, nd in zip(raw_area_ratios, new_decisions) if nd]
            average_motion_area = (sum(retained_ratios) / len(retained_ratios)) if retained_ratios else 0.0
            
            sweep_results.append({
                "video_name": video_name,
                "method": method,
                "threshold_percentage": f"{t*100:.1f}%",
                "frames_retained": frames_retained,
                "reduction_percentage": reduction_percentage,
                "continuity_score": continuity_score,
                "average_motion_area": average_motion_area
            })
            
    # Save search results to CSV
    csv_path = os.path.join(output_dir, "motion_area_threshold_search.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["video_name", "method", "threshold_percentage", "frames_retained", "reduction_percentage", "continuity_score", "average_motion_area"]
        )
        writer.writeheader()
        for sr in sweep_results:
            writer.writerow(sr)
            
    print(f"Saved threshold search CSV to: {csv_path}")
    
    # Generate graphs: outputs/experiments/plots/motion_area_threshold.png
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # We will average the results across all videos for plotting
    methods = sorted(list(set(sr["method"] for sr in sweep_results)))
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Reduction % vs Threshold %
    plt.subplot(1, 2, 1)
    for m in methods:
        method_runs = [sr for sr in sweep_results if sr["method"] == m]
        # Average per threshold percentage
        t_labels = [f"{t*100:.1f}%" for t in thresholds]
        t_reductions = []
        for tl in t_labels:
            runs_at_t = [run for run in method_runs if run["threshold_percentage"] == tl]
            t_reductions.append(np.mean([run["reduction_percentage"] for run in runs_at_t]) if runs_at_t else 0.0)
            
        plt.plot(t_labels, t_reductions, marker='o', label=m)
    plt.title("Threshold vs Frame Reduction %", fontsize=11, fontweight='bold')
    plt.xlabel("Motion Area Threshold", fontsize=9)
    plt.ylabel("Reduction Percentage (%)", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Subplot 2: Continuity Score vs Threshold %
    plt.subplot(1, 2, 2)
    for m in methods:
        method_runs = [sr for sr in sweep_results if sr["method"] == m]
        t_labels = [f"{t*100:.1f}%" for t in thresholds]
        t_continuities = []
        for tl in t_labels:
            runs_at_t = [run for run in method_runs if run["threshold_percentage"] == tl]
            t_continuities.append(np.mean([run["continuity_score"] for run in runs_at_t]) if runs_at_t else 0.0)
            
        plt.plot(t_labels, t_continuities, marker='s', label=m)
    plt.title("Threshold vs Continuity Score", fontsize=11, fontweight='bold')
    plt.xlabel("Motion Area Threshold", fontsize=9)
    plt.ylabel("Continuity Score (0-1)", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "motion_area_threshold.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved threshold sweep plot to: {plot_path}")

def clean_temp_dirs(output_dir):
    """Remove temporary visualizer folders created by evaluator."""
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Batch Motion Detection Experiment Infrastructure")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/videos",
        help="Path to dataset containing CCTV videos (default: datasets/videos)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Directory to save csv results and plots (default: outputs/experiments)"
    )
    parser.add_argument(
        "--param-search",
        action="store_true",
        help="Run optional Grid Search parameter testing on Frame Difference detector"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-discover or instantiate sample video
    videos = ensure_sample_video(args.dataset)
    
    if not videos:
        print("Error: No videos to evaluate. Exiting.")
        sys.exit(1)
        
    print(f"Discovered {len(videos)} video files for batch evaluation.")
    
    # 1. Run Standard Batch Evaluation
    results = run_standard_experiments(videos, args.output_dir)
    
    # 2. Generate aggregate graphs
    plots_dir = os.path.join(args.output_dir, "plots")
    if results:
        generate_plots(results, plots_dir)
        # Run area threshold analysis sweep
        run_motion_area_threshold_sweep(results, args.output_dir)
        
    # 3. Optional Grid Search Sweep
    if args.param_search:
        run_parameter_grid_search(videos, args.output_dir)
        
    # Clean temporary evaluator artifacts
    clean_temp_dirs(args.output_dir)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT INFRASTRUCTURE BATCH RUN COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

import os
import time
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.motion.base import MotionDetector

class MotionBenchmarkEvaluator:
    """
    Runner for benchmarking different motion detection techniques on a target video.
    """

    def __init__(self, video_path: str, output_dir: str = "outputs") -> None:
        """
        Initialize the evaluator.
        
        Args:
            video_path: Path to the target video.
            output_dir: Root directory to save outputs.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "motion_results")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_detector(
        self,
        name: str,
        detector: MotionDetector,
        sample_frame_indices: list[int] = None,
        max_video_frames: int = 500,
        save_viz: bool = True
    ) -> dict:
        """
        Evaluate a single detector on the video.
        
        Args:
            name: Display name of the detector.
            detector: The MotionDetector instance.
            sample_frame_indices: Frame indices at which to save sample mask images.
            max_video_frames: Max frames to write to the sample processed video.
            save_viz: If True, saves visualization videos and masks.
            
        Returns:
            Dictionary containing metrics.
        """
        if sample_frame_indices is None:
            sample_frame_indices = [50, 100, 150, 200]

        is_webcam = str(self.video_path).isdigit()
        video_source = int(self.video_path) if is_webcam else self.video_path
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_path}")

        total_frames = 0
        motion_frames = 0
        motion_decisions = []
        motion_area_ratios = []
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 20.0

        # Video writer for a short processed video showing detected motion (side-by-side)
        video_writer = None
        if save_viz:
            video_filename = os.path.join(self.results_dir, f"{name}_motion.mp4")
            
            # We will resize to a standard height of 360px for the side-by-side video to keep it compact
            viz_h = 360
            viz_w = int(width * (viz_h / height))
            
            # Side-by-side dimensions: width is viz_w * 2, height is viz_h
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (viz_w * 2, viz_h))

        print(f"Running benchmark for: {name}...")
        
        start_time = time.perf_counter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            if is_webcam and total_frames > max_video_frames:
                break
                
            # Run motion detector
            motion_detected, mask = detector.process(frame)
            motion_decisions.append(motion_detected)
            
            # Calculate motion area ratio
            h_mask, w_mask = mask.shape[:2]
            total_mask_pixels = h_mask * w_mask
            motion_area_ratio = (cv2.countNonZero(mask) / total_mask_pixels) if total_mask_pixels > 0 else 0.0
            motion_area_ratios.append(motion_area_ratio)
            
            if motion_detected:
                motion_frames += 1
                
            if save_viz:
                # Save sample masks at designated frame numbers
                if total_frames in sample_frame_indices:
                    mask_filename = os.path.join(self.results_dir, f"mask_{name}_frame_{total_frames}.png")
                    cv2.imwrite(mask_filename, mask)
                    
                # Write a short side-by-side video (up to max_video_frames total frames)
                if total_frames <= max_video_frames and video_writer is not None and video_writer.isOpened():
                    frame_res = cv2.resize(frame, (viz_w, viz_h))
                    mask_res = cv2.resize(mask, (viz_w, viz_h))
                    mask_bgr = cv2.cvtColor(mask_res, cv2.COLOR_GRAY2BGR)
                    
                    # If motion detected, draw a red border or label on the original frame
                    if motion_detected:
                        cv2.rectangle(frame_res, (0, 0), (viz_w - 1, viz_h - 1), (0, 0, 255), 4)
                        cv2.putText(frame_res, "MOTION DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame_res, "NO MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    # Concatenate horizontally
                    sbs_frame = cv2.hconcat([frame_res, mask_bgr])
                    video_writer.write(sbs_frame)

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        cap.release()
        if video_writer is not None:
            video_writer.release()
            
        discarded_frames = total_frames - motion_frames
        reduction_percentage = (discarded_frames / total_frames * 100) if total_frames > 0 else 0.0
        
        # Calculate motion quality metrics
        total_motion = sum(motion_decisions)
        
        # 1. Motion Continuity Score: consecutive transitions / total motion frames
        consecutive_motion = 0
        for i in range(1, len(motion_decisions)):
            if motion_decisions[i] and motion_decisions[i-1]:
                consecutive_motion += 1
        continuity_score = (consecutive_motion / total_motion) if total_motion > 0 else 0.0
        
        # 2. Number of motion segments and Average motion segment length
        segments = []
        current_len = 0
        for m in motion_decisions:
            if m:
                current_len += 1
            else:
                if current_len > 0:
                    segments.append(current_len)
                    current_len = 0
        if current_len > 0:
            segments.append(current_len)
            
        num_segments = len(segments)
        avg_segment_length = (sum(segments) / num_segments) if num_segments > 0 else 0.0
        
        # Compute motion area ratio metrics for frames where motion is detected
        motion_ratios_where_detected = [
            r for r, d in zip(motion_area_ratios, motion_decisions) if d
        ]
        if motion_ratios_where_detected:
            average_motion_area_ratio = sum(motion_ratios_where_detected) / len(motion_ratios_where_detected)
            maximum_motion_area_ratio = max(motion_ratios_where_detected)
            minimum_motion_area_ratio = min(motion_ratios_where_detected)
        else:
            average_motion_area_ratio = 0.0
            maximum_motion_area_ratio = 0.0
            minimum_motion_area_ratio = 0.0
            
        print(f"Finished {name}. Processed {total_frames} frames in {processing_time:.2f}s. Reduction: {reduction_percentage:.1f}%")
        
        return {
            "method": name,
            "total_frames": total_frames,
            "motion_frames": motion_frames,
            "discarded_frames": discarded_frames,
            "reduction_percentage": reduction_percentage,
            "time_seconds": processing_time,
            "continuity_score": continuity_score,
            "avg_segment_length": avg_segment_length,
            "num_segments": num_segments,
            "average_motion_area_ratio": average_motion_area_ratio,
            "maximum_motion_area_ratio": maximum_motion_area_ratio,
            "minimum_motion_area_ratio": minimum_motion_area_ratio,
            "motion_area_ratios": motion_area_ratios,
            "motion_decisions": motion_decisions
        }

    def run_all(self, detectors: dict[str, MotionDetector], save_viz: bool = True) -> list[dict]:
        """
        Run the benchmark for a list of detectors, save the CSV, and generate comparison graphs.
        
        Args:
            detectors: A dictionary mapping detector names to their respective instances.
            save_viz: If True, saves video and mask visualization files.
            
        Returns:
            List of results dictionaries.
        """
        results = []
        for name, detector in detectors.items():
            res = self.evaluate_detector(name, detector, save_viz=save_viz)
            results.append(res)
            
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "motion_benchmark.csv")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=[
                    "method", "total_frames", "motion_frames", "discarded_frames", 
                    "reduction_percentage", "time_seconds", "continuity_score", 
                    "avg_segment_length", "num_segments", "average_motion_area_ratio", 
                    "maximum_motion_area_ratio", "minimum_motion_area_ratio"
                ]
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "method": r["method"],
                    "total_frames": r["total_frames"],
                    "motion_frames": r["motion_frames"],
                    "discarded_frames": r["discarded_frames"],
                    "reduction_percentage": r["reduction_percentage"],
                    "time_seconds": r["time_seconds"],
                    "continuity_score": r["continuity_score"],
                    "avg_segment_length": r["avg_segment_length"],
                    "num_segments": r["num_segments"],
                    "average_motion_area_ratio": r["average_motion_area_ratio"],
                    "maximum_motion_area_ratio": r["maximum_motion_area_ratio"],
                    "minimum_motion_area_ratio": r["minimum_motion_area_ratio"]
                })
                
        print(f"Saved benchmark results to: {csv_path}")
        
        # Generate graphs
        self._generate_graphs(results)
        
        return results

    def _generate_graphs(self, results: list[dict]) -> None:
        """Helper to generate comparison plots using matplotlib."""
        methods = [r["method"] for r in results]
        reductions = [r["reduction_percentage"] for r in results]
        times = [r["time_seconds"] for r in results]
        
        # Plot 1: Reduction comparison
        plt.figure(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
        bars = plt.bar(methods, reductions, color=colors, edgecolor='grey')
        plt.title("Frame Reduction Percentage by Method", fontsize=14, fontweight='bold')
        plt.xlabel("Motion Detection Method", fontsize=11)
        plt.ylabel("Reduction (%)", fontsize=11)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        reduction_plot_path = os.path.join(self.results_dir, "reduction_comparison.png")
        plt.savefig(reduction_plot_path, dpi=150)
        plt.close()
        print(f"Saved reduction comparison plot to: {reduction_plot_path}")

        # Plot 2: Speed comparison
        plt.figure(figsize=(8, 5))
        bars = plt.bar(methods, times, color=colors, edgecolor='grey')
        plt.title("Execution Time Comparison", fontsize=14, fontweight='bold')
        plt.xlabel("Motion Detection Method", fontsize=11)
        plt.ylabel("Execution Time (seconds)", fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(times)*0.02), f"{height:.3f}s", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        speed_plot_path = os.path.join(self.results_dir, "speed_comparison.png")
        plt.savefig(speed_plot_path, dpi=150)
        plt.close()
        print(f"Saved speed comparison plot to: {speed_plot_path}")

"""Batch Evaluation Executable for Progressive Search-Space Analysis.

Scans dataset video files, runs instrumented pipeline statistics collection,
saves JSON/CSV statistics, generates research plots, and produces pipeline_report.md.
"""

import os
import sys
import glob
import time
import cv2
import argparse
from typing import List

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.detector import Detector
from src.pipeline.tracking_stage import TrackingStage
from src.pipeline.relationship_stage import RelationshipStage
from src.pipeline.pipeline import Pipeline
from src.core.models.frame_context import FrameContext
from src.evaluation.pipeline_statistics import (
    PipelineStatisticsCollector,
    save_pipeline_statistics,
    generate_sankey_data,
    generate_research_plots,
    generate_pipeline_report
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


def run_pipeline_for_video(video_path: str, detector: Detector, tracking_stage: TrackingStage, relationship_stage: RelationshipStage) -> dict:
    """Run pipeline and measure statistics per stage for a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0

    collector = PipelineStatisticsCollector(video_path, fps=fps, total_video_frames=total_frames)
    collector.log_input(total_frames)

    # Re-instantiate MOG2 and tracker for clean video state
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    tracking_stage.tracker = tracking_stage.tracker.__class__()
    tracking_stage.history_manager.clear()

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        current_ts = frame_num / fps

        # -------------------------------------------------------------
        # Stage 1: Motion Filtering
        # -------------------------------------------------------------
        t0 = time.time()
        fg_mask = mog2.apply(frame)
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_retained = (motion_pixels > 5000)
        t_motion = time.time() - t0

        collector.record_motion_stage(motion_retained, t_motion)

        if not motion_retained:
            collector.record_yolo_stage([], 0.0)
            collector.record_tracking_stage([], 0.0)
            collector.record_relationship_stage([], 0, 0, 0.0)
            collector.record_candidate_event_stage(False, 0.0)
            continue

        # -------------------------------------------------------------
        # Stage 2: YOLO Detection
        # -------------------------------------------------------------
        t0 = time.time()
        detections = detector.detect(frame)
        t_yolo = time.time() - t0

        collector.record_yolo_stage(detections, t_yolo)

        if not detections:
            collector.record_tracking_stage([], 0.0)
            collector.record_relationship_stage([], 0, 0, 0.0)
            collector.record_candidate_event_stage(False, 0.0)
            continue

        # -------------------------------------------------------------
        # Stage 3: Tracking
        # -------------------------------------------------------------
        context = FrameContext(
            frame=frame,
            frame_number=frame_num,
            timestamp=current_ts,
            detections=detections
        )

        t0 = time.time()
        context = tracking_stage.process(context)
        t_tracking = time.time() - t0

        tracks = context.tracks
        collector.record_tracking_stage(tracks, t_tracking)

        if not tracks:
            collector.record_relationship_stage([], 0, 0, 0.0)
            collector.record_candidate_event_stage(False, 0.0)
            continue

        # -------------------------------------------------------------
        # Stage 4: Relationship Engine
        # -------------------------------------------------------------
        num_persons = sum(1 for trk in tracks if getattr(trk, 'class_name', '') == 'person')
        num_vehicles = sum(1 for trk in tracks if getattr(trk, 'class_name', '') in {"bicycle", "motorcycle", "car", "bus", "truck"})

        t0 = time.time()
        context = relationship_stage.process(context)
        t_rel = time.time() - t0

        relationships = context.metadata.get("relationships", [])
        collector.record_relationship_stage(relationships, num_persons, num_vehicles, t_rel)

        # -------------------------------------------------------------
        # Stage 5: Candidate Event Generation
        # -------------------------------------------------------------
        t0 = time.time()
        # Candidate event defined as presence of valid proximity events
        is_candidate_event = len(relationships) > 0
        t_event = time.time() - t0

        collector.record_candidate_event_stage(is_candidate_event, t_event)

    cap.release()
    return collector.finalize()


def main():
    parser = argparse.ArgumentParser(description="AI Forensic Search - Progressive Pipeline Statistics Evaluator")
    parser.add_argument("--dataset", nargs="+", default=["Snatch 1.0", "datasets/videos"], help="Paths to dataset directories")
    parser.add_argument("--output_dir", type=str, default="outputs/pipeline_statistics", help="Output directory for reports and plots")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos for fast benchmarking")

    args = parser.parse_args()

    print(f"Discovering video files in: {args.dataset}...")
    videos = discover_videos(args.dataset)
    print(f"Found {len(videos)} video files.")

    if not videos:
        # Fallback check for outputs/motion_output.avi or webcam_recording.mp4
        fallback = os.path.abspath("outputs/webcam_recording.mp4")
        if os.path.exists(fallback):
            videos = [fallback]
            print(f"Using fallback video: {fallback}")
        else:
            print("No video files found. Please provide a valid dataset path.")
            sys.exit(1)

    if args.limit:
        videos = videos[:args.limit]
        print(f"Limited execution to first {len(videos)} videos.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize shared detector modules
    print("Initializing YOLO detector and pipeline stages...")
    detector = Detector()
    tracking_stage = TrackingStage()
    relationship_stage = RelationshipStage(distance_threshold=150.0)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    all_stats = []

    print("\nExecuting Progressive Forensic Search-Space Statistics Collection...")
    print("=" * 70)

    for idx, video_path in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Processing: {os.path.basename(video_path)} ...")
        st = run_pipeline_for_video(video_path, detector, tracking_stage, relationship_stage)
        if st is not None:
            all_stats.append(st)
            rem = st["pipeline_summary"]["search_space_remaining_pct"]["Candidate Events"]
            t_sec = st["pipeline_summary"]["total_execution_time_seconds"]
            print(f"    └─ Completed in {t_sec:.2f}s | Search Space Remaining: {rem:.2f}%")

    print("\n" + "=" * 70)
    print("Saving Dataset Pipeline Statistics & Generating Research Reports...")
    print("=" * 70)

    # Task 2: Save JSON and CSV
    save_pipeline_statistics(all_stats, args.output_dir)
    print(f" Saved: {os.path.join(args.output_dir, 'pipeline_statistics.json')}")
    print(f" Saved: {os.path.join(args.output_dir, 'pipeline_statistics.csv')}")

    # Task 3: Visualizations & Sankey data
    generate_sankey_data(all_stats, args.output_dir)
    print(f" Saved: {os.path.join(args.output_dir, 'pipeline_sankey_data.csv')}")

    generate_research_plots(all_stats, args.output_dir)
    print(f" Saved: {os.path.join(args.output_dir, 'search_space_reduction.png')}")
    print(f" Saved: {os.path.join(args.output_dir, 'stage_runtime.png')}")
    print(f" Saved: {os.path.join(args.output_dir, 'detection_distribution.png')}")

    # Task 5: Research Report
    generate_pipeline_report(all_stats, args.output_dir)
    print(f" Saved: {os.path.join(args.output_dir, 'pipeline_report.md')}")

    print("\nPipeline Statistics Benchmark Complete successfully!")


if __name__ == "__main__":
    main()

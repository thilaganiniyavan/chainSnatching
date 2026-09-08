"""End-to-End Evaluation Application Runner Script.

Executes full multi-stage pipeline benchmarks across CCTV video files in an input directory.
Measures stage-wise latency, frame reduction cascade, hardware resource usage, and evidence yield.

Outputs:
- pipeline_statistics.csv
- stage_statistics.csv
- runtime_statistics.csv
- system_resource_usage.csv
- framework_summary.md
- Publication figures (outputs/evaluation_results/figures/)
"""

import argparse
import gc
import glob
import os
import sys
import time

import cv2

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection.detector import Detector
from src.pipeline.pipeline import Pipeline
from src.pipeline.tracking_stage import TrackingStage
from src.pipeline.relationship_stage import RelationshipStage
from src.pipeline.interaction_stage import InteractionStage
from src.pipeline.behaviour_stage import BehaviourStage
from src.pipeline.reasoning_stage import ReasoningStage
from src.pipeline.graph_reasoning_stage import GraphReasoningStage
from src.pipeline.roi_selection_stage import ROISelectionStage
from src.pipeline.pose_estimation_stage import PoseEstimationStage
from src.pipeline.skeleton_sequence_stage import SkeletonSequenceStage
from src.pipeline.action_recognition_stage import ActionRecognitionStage
from src.pipeline.behaviour_fusion_stage import BehaviourFusionStage
from src.pipeline.snatch_signature_stage import SnatchSignatureStage
from src.pipeline.forensic_indexing_stage import ForensicIndexingStage
from src.core.models.frame_context import FrameContext
from src.evaluation.pipeline_evaluator import PipelineEvaluator, STAGE_NAMES


def run_evaluation_on_video(
    video_path: str,
    args: argparse.Namespace,
    evaluator: PipelineEvaluator,
    detector: Detector,
    video_idx: int = 1,
    total_videos: int = 1,
) -> None:
    """Run full pipeline on a single video file and record evaluation metrics."""
    video_basename = os.path.basename(video_path)
    print(f"\n[{video_idx}/{total_videos}] Evaluating: {video_basename} ...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ! Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 20.0

    max_limit = args.max_frames if (args.max_frames and args.max_frames > 0) else total_frames
    target_frames = min(total_frames, max_limit) if (max_limit and total_frames > 0) else max_limit

    # Initialize background subtractor
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Initialize all 13 pipeline stages
    tracking_stage = TrackingStage()
    relationship_stage = RelationshipStage(distance_threshold=150.0)
    interaction_stage = InteractionStage(distance_threshold=150.0)
    behaviour_stage = BehaviourStage(fps=fps)
    reasoning_stage = ReasoningStage(fps=fps)
    graph_stage = GraphReasoningStage(fps=fps)
    roi_stage = ROISelectionStage(fps=fps)
    pose_stage = PoseEstimationStage(backend_name=args.backend, fps=fps)
    sequence_stage = SkeletonSequenceStage(normalization_method=args.norm, fps=fps)
    action_stage = ActionRecognitionStage(backend_name=args.action_backend, fps=fps)
    fusion_stage = BehaviourFusionStage(fusion_strategy=args.fusion_strategy, fps=fps)
    snatch_stage = SnatchSignatureStage()
    indexing_stage = ForensicIndexingStage(video_id=video_basename, location="Camera 1")

    stages = [
        tracking_stage,
        relationship_stage,
        interaction_stage,
        behaviour_stage,
        reasoning_stage,
        graph_stage,
        roi_stage,
        pose_stage,
        sequence_stage,
        action_stage,
        fusion_stage,
        snatch_stage,
        indexing_stage,
    ]
    pipeline = Pipeline(stages=stages)

    frame_number = 0
    motion_triaged_cnt = 0
    processed_cnt = 0
    last_context = None

    start_time = time.time()
    stage_times: dict[str, float] = {name: 0.0 for name in STAGE_NAMES}
    stage_map = {
        "TrackingStage": "Multi-Object Tracking",
        "RelationshipStage": "Relationship Engine",
        "InteractionStage": "Interaction Manager",
        "BehaviourStage": "Behaviour Intelligence",
        "ReasoningStage": "Behaviour Intelligence",
        "GraphReasoningStage": "Behaviour Graph Reasoning",
        "ROISelectionStage": "ROI Selection",
        "PoseEstimationStage": "Pose Estimation",
        "SkeletonSequenceStage": "Skeleton Sequence Builder",
        "ActionRecognitionStage": "Human Action Recognition",
        "BehaviourFusionStage": "Behaviour Fusion",
        "SnatchSignatureStage": "Snatch Signature Engine",
        "ForensicIndexingStage": "Forensic Indexing & Retrieval",
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if args.max_frames and frame_number > args.max_frames:
                break

            # Motion Triage
            t0 = time.perf_counter()
            fg_mask = mog2.apply(frame)
            motion_pixels = cv2.countNonZero(fg_mask)
            stage_times["Motion Triage"] += (time.perf_counter() - t0) * 1000.0

            if motion_pixels <= 5000:
                continue

            motion_triaged_cnt += 1

            # Semantic Filtering & Detection
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            det_elapsed = (time.perf_counter() - t0) * 1000.0
            stage_times["YOLO Detection"] += det_elapsed

            sf_t0 = time.perf_counter()
            has_detections = bool(detections)
            stage_times["Semantic Filtering"] += (time.perf_counter() - sf_t0) * 1000.0

            if not has_detections:
                continue

            processed_cnt += 1

            context = FrameContext(
                frame=frame,
                frame_number=frame_number,
                timestamp=frame_number / fps,
                detections=detections,
            )

            # Measure each downstream stage individually
            for stage in pipeline.stages:
                s_t0 = time.perf_counter()
                context = stage.process(context)
                s_elapsed = (time.perf_counter() - s_t0) * 1000.0
                stage_key = stage_map.get(type(stage).__name__)
                if stage_key in stage_times:
                    stage_times[stage_key] += s_elapsed

            last_context = context

            if frame_number % 10 == 0 or frame_number == target_frames:
                print(f"  Frame {frame_number}/{target_frames} (Processed: {processed_cnt})...", end="\r", flush=True)

    finally:
        cap.release()

        # Finalize stages safely
        for stage in stages:
            if hasattr(stage, "finalize"):
                try:
                    stage.finalize()
                except Exception:
                    pass

        # Explicitly release MediaPipe C++ session
        if hasattr(pose_stage, "estimator"):
            est = pose_stage.estimator
            if hasattr(est, "_pose_solution") and est._pose_solution is not None:
                try:
                    est._pose_solution.close()
                    est._pose_solution = None
                except Exception:
                    pass

    elapsed_seconds = max(1e-6, time.time() - start_time)
    fps_metric = processed_cnt / elapsed_seconds if processed_cnt > 0 else 0.0

    # Extract artifact counts safely
    det_cnt = len(last_context.detections) if (last_context and last_context.detections) else 0
    tr_cnt = len(last_context.tracks) if (last_context and hasattr(last_context, "tracks")) else 0
    int_cnt = len(last_context.interactions) if (last_context and hasattr(last_context, "interactions")) else 0
    gr_cnt = len(graph_stage.engine.get_completed_graphs())
    roi_cnt = len(roi_stage.engine.get_accepted_rois())
    pose_cnt = len(pose_stage.logger.get_pose_results())
    seq_cnt = len(sequence_stage.builder.get_completed_sequences())
    act_cnt = len(action_stage.logger.get_results())
    fus_cnt = len(fusion_stage.engine.get_completed_fusions())
    sig_cnt = len(snatch_stage.engine.get_all_results())
    evt_cnt = len(indexing_stage.query_engine.get_all_events())

    evaluator.evaluate_video(
        video_path=video_path,
        total_frames=max(total_frames, frame_number),
        processed_frames=processed_cnt,
        motion_triaged_frames=motion_triaged_cnt,
        detection_cnt=det_cnt,
        track_cnt=tr_cnt,
        interaction_cnt=int_cnt,
        graph_cnt=gr_cnt,
        roi_cnt=roi_cnt,
        pose_cnt=pose_cnt,
        sequence_cnt=seq_cnt,
        action_cnt=act_cnt,
        fusion_cnt=fus_cnt,
        signature_cnt=sig_cnt,
        forensic_event_cnt=evt_cnt,
        elapsed_seconds=elapsed_seconds,
        stage_times_ms=stage_times,
    )

    print(f"\n  ✓ Done {video_basename}: {processed_cnt}/{frame_number} frames processed in {elapsed_seconds:.1f}s ({fps_metric:.1f} FPS)\n", flush=True)

    # Immediate garbage collection between videos
    del pipeline, stages, mog2, last_context
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Evaluation Runner")
    parser.add_argument("--input-dir", type=str, help="Directory containing CCTV video files")
    parser.add_argument("--input", type=str, help="Single CCTV video file path")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation_results", help="Directory for evaluation results")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to evaluate (e.g. 5 or 10)")
    parser.add_argument("--max-frames", type=int, default=350, help="Maximum frames per video to evaluate (default: 350, pass 0 for all frames)")
    parser.add_argument("--backend", type=str, default="mediapipe", help="Pose estimation backend")
    parser.add_argument("--norm", type=str, default="hip_centered", help="Skeleton normalization strategy")
    parser.add_argument("--action-backend", type=str, default="stgcn", help="Action recognition backend")
    parser.add_argument("--fusion-strategy", type=str, default="weighted_confidence", help="Evidence fusion strategy")
    args = parser.parse_args()

    evaluator = PipelineEvaluator(output_dir=args.output_dir)

    video_files = []
    if args.input_dir and os.path.exists(args.input_dir):
        video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")) + glob.glob(os.path.join(args.input_dir, "*.avi")))
    elif args.input and os.path.exists(args.input):
        video_files = [args.input]

    if not video_files:
        print(f"No video files found in '{args.input_dir or args.input}'. Evaluation cannot proceed.")
        return

    if args.max_videos and len(video_files) > args.max_videos:
        print(f"Limiting evaluation to a sample of {args.max_videos} video file(s)...")
        video_files = video_files[:args.max_videos]

    print(f"\nStarting End-to-End Framework Evaluation across {len(video_files)} video file(s)...")
    print(f"Max frames per video: {args.max_frames if args.max_frames else 'All'}\n")

    # Reuse detector across videos
    detector = Detector()

    for idx, v_path in enumerate(video_files, start=1):
        run_evaluation_on_video(v_path, args, evaluator, detector, video_idx=idx, total_videos=len(video_files))

    evaluator.export_all()

    print("\n============================================================")
    print("End-to-End Pipeline Evaluation Completed Successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Thesis Report: {os.path.join(args.output_dir, 'framework_summary.md')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()

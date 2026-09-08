"""Diagnostic Inspection Tool for ROIs, Pose Skeletons, and Action Recognition.

Processes snatch video clips (or a specified video) and prints detailed outputs for:
1. ROI Selection Stage (Bounding boxes, frame spans, quality scores, acceptance)
2. Pose Estimation Stage (MediaPipe 33 keypoints, joints detected, confidence)
3. Skeleton Sequence Stage (Tensors constructed, sequence lengths, frame indices)
4. ST-GCN Action Recognition Stage (Predicted action labels, top-k probabilities, confidence)
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.pipeline.pipeline import Pipeline
from src.core.models.frame_context import FrameContext
from src.evaluation.research_live_evaluator import (
    LiveResearchEvaluator,
    full_spec,
    discover_snatch_videos,
    is_incident_video,
    _finalize_stage,
    MOTION_PIXEL_THRESHOLD,
)


def inspect_video(evaluator: LiveResearchEvaluator, spec, video_path: str, max_frames: int = 1000, export_video: bool = True, output_dir: str = "outputs/inspected_videos"):
    vname = os.path.basename(video_path)
    gt = "SNATCH THEFT" if is_incident_video(video_path) else "NORMAL"

    print("\n" + "=" * 80)
    print(f" DEEP INSPECTION: {vname} (Ground Truth: {gt})")
    print("=" * 80)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    if fps <= 0 or fps != fps:
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    video_writer = None
    out_video_path = None
    frames_written = 0
    if export_video:
        os.makedirs(output_dir, exist_ok=True)
        out_video_name = f"{os.path.splitext(vname)[0]}_inspected.mp4"
        out_video_path = os.path.join(output_dir, out_video_name)
        
        # Try mp4v first, fallback to XVID or MJPG
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    detector = evaluator._ensure_detector()
    stages, handles = evaluator._build_stages(spec, fps=fps, video_name=vname)
    pipeline = Pipeline(stages=stages)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    frame_number = 0
    processed_cnt = 0

    all_poses_collected = []
    all_actions_collected = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            if max_frames and frame_number > max_frames:
                break

            if mog2 is not None:
                mask = mog2.apply(frame)
                if cv2.countNonZero(mask) <= MOTION_PIXEL_THRESHOLD:
                    if video_writer and video_writer.isOpened():
                        video_writer.write(frame)
                        frames_written += 1
                    continue

            detections = detector.detect(frame)
            if not detections:
                if video_writer and video_writer.isOpened():
                    video_writer.write(frame)
                    frames_written += 1
                continue

            processed_cnt += 1
            context = FrameContext(
                frame=frame,
                frame_number=frame_number,
                timestamp=frame_number / fps,
                detections=detections,
            )
            pipeline.run(context)

            # Write annotated frame with ROI, Skeletons, and Actions
            if video_writer and video_writer.isOpened():
                vis_frame = context.metadata.get(
                    "signature_frame",
                    context.metadata.get(
                        "action_frame",
                        context.metadata.get(
                            "sequence_frame",
                            context.metadata.get(
                                "pose_frame",
                                context.metadata.get(
                                    "roi_frame",
                                    context.frame
                                )
                            )
                        )
                    )
                )
                if vis_frame is not None:
                    if vis_frame.shape[:2] != (height, width):
                        vis_frame = cv2.resize(vis_frame, (width, height))
                    video_writer.write(vis_frame)
                else:
                    video_writer.write(frame)
                frames_written += 1

            # Collect frame outputs
            poses = context.metadata.get("pose_results", [])
            actions = context.metadata.get("action_results", [])
            if poses:
                all_poses_collected.extend(poses)
            if actions:
                all_actions_collected.extend(actions)

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            if frames_written > 0:
                print(f"\n  🎥 Annotated Inspection Video Saved ({frames_written} frames): {os.path.abspath(out_video_path)}")
            else:
                print(f"\n  ! Warning: VideoWriter opened but 0 frames written.")
        for stage in stages:
            _finalize_stage(stage)

    # 1. ROI Inspection
    roi_stage = handles.get("roi")
    rois = roi_stage.engine.get_completed_rois() if roi_stage else []
    accepted_rois = roi_stage.engine.get_accepted_rois() if roi_stage else []

    print(f"\n[1] ROI SELECTION STAGE (Total ROIs: {len(rois)} | Accepted: {len(accepted_rois)})")
    print("-" * 80)
    if not rois:
        print("  ! No ROIs were generated for this video.")
    else:
        for idx, r in enumerate(rois, start=1):
            status = "ACCEPTED ✓" if r.is_accepted else "REJECTED ✗"
            print(f"  ROI #{idx}: ID={r.roi_id} | Interaction={r.interaction_id} | Status={status}")
            print(f"    • Frames Span: Frame {r.start_frame} to {r.end_frame} (Total {r.frame_count} frames | {r.duration_seconds:.2f}s)")
            print(f"    • Track Targets: Person Track #{r.person_track_id}, Vehicle Track #{r.vehicle_track_id}")
            if r.quality_metrics:
                qm = r.quality_metrics
                print(f"    • Quality: Completeness={qm.get('completeness', 0.0):.2f}, Missing%={qm.get('missing_percentage', 0.0):.1f}%, Stability={qm.get('stability', 0.0):.2f}")
            if not r.is_accepted:
                print(f"    • Rejection Reason: {r.rejection_reason}")

    # 2. Pose Estimation Inspection
    pose_stage = handles.get("pose")
    print(f"\n[2] POSE ESTIMATION STAGE (Total Pose Frames Logged: {len(all_poses_collected)})")
    print("-" * 80)
    if not all_poses_collected:
        print("  ! No human poses were estimated (MediaPipe returned 0 keypoints).")
    else:
        sample_poses = all_poses_collected[:8]
        for p in sample_poses:
            valid_joints = sum(1 for kp in p.keypoints_pixel if kp[2] > 0.3)
            print(f"  Pose Frame #{p.frame_index}: Track #{p.track_id} | Valid Keypoints: {valid_joints}/{p.num_keypoints} ({p.topology}) | Mean Conf: {p.overall_confidence:.3f}")

    # 3. Skeleton Sequence Inspection
    seq_stage = handles.get("sequence")
    seqs = seq_stage.builder.get_completed_sequences() if seq_stage else []
    print(f"\n[3] SKELETON SEQUENCE STAGE (Total Sequences Built: {len(seqs)})")
    print("-" * 80)
    if not seqs:
        print("  ! No SkeletonSequences constructed.")
    else:
        for s in seqs:
            status = "ACCEPTED ✓" if s.is_accepted else "REJECTED ✗"
            print(f"  Sequence ID: {s.sequence_id} | Track #{s.person_track_id} | Status={status}")
            print(f"    • Length: {s.frame_count} frames (Frames {s.start_frame}–{s.end_frame})")
            print(f"    • Quality Score: {s.quality_score:.3f} | Completeness: {s.completeness_score:.3f}")
            if not s.is_accepted:
                print(f"    • Rejection Reason: {s.rejection_reason}")

    # 4. Action Recognition Inspection
    act_stage = handles.get("action")
    print(f"\n[4] ST-GCN ACTION RECOGNITION STAGE (Total Classified Actions: {len(all_actions_collected)})")
    print("-" * 80)
    if not all_actions_collected:
        print("  ! No action recognitions completed.")
    else:
        for a in all_actions_collected:
            print(f"  Action Result ID: {a.action_id} | Track #{a.track_id}")
            print(f"    • Primary Prediction : '{a.predicted_action}' (Confidence: {a.action_confidence:.4f})")
            if a.top_k_predictions:
                top3_str = ", ".join([f"{lbl}: {prob:.3f}" for lbl, prob in a.top_k_predictions[:4]])
                print(f"    • Top Predictions    : {top3_str}")
            elif a.class_probabilities:
                sorted_probs = sorted(a.class_probabilities.items(), key=lambda x: -x[1])[:4]
                top3_str = ", ".join([f"{lbl}: {prob:.3f}" for lbl, prob in sorted_probs])
                print(f"    • Top Predictions    : {top3_str}")

    # 5. Snatch Signature Output
    sig_stage = handles.get("signature")
    results = sig_stage.engine.get_all_results() if sig_stage else []
    print(f"\n[5] SNATCH SIGNATURE MATCHING (Total Candidate Signatures: {len(results)})")
    print("-" * 80)
    for r in results:
        matched = [m["component"] for m in r.matched_evidence]
        missing = [m["component"] for m in r.missing_evidence]
        print(f"  Signature Score: {r.signature_score:.4f} | Decision: {r.decision}")
        print(f"    • Matched Evidence: {matched}")
        print(f"    • Missing Evidence: {missing}")
        print(f"    • Actions Timeline: {r.action_evidence}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect ROIs, Pose Skeletons, and Actions on Snatch Videos")
    parser.add_argument("--video", type=str, default=None, help="Specific video path (e.g. 'Snatch 1.0/Chain Snatching Videos/Snatch Theft/1.mp4')")
    parser.add_argument("--max-videos", type=int, default=2, help="Number of snatch videos to inspect (default: 2)")
    parser.add_argument("--max-frames", type=int, default=1000, help="Max frames to process per video (default: 1000)")
    parser.add_argument("--export-video", action="store_true", default=True, help="Export visual inspection MP4 videos with ROI, Pose, and Action overlays (default: True)")
    parser.add_argument("--output-dir", type=str, default="outputs/inspected_videos", help="Directory to save inspection videos")
    args = parser.parse_args()

    evaluator = LiveResearchEvaluator(max_frames=args.max_frames)
    spec = full_spec("Config D (Proposed Framework)")

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: File not found: {args.video}")
            return
        video_list = [os.path.abspath(args.video)]
    else:
        all_videos = discover_snatch_videos()
        pos = [v for v in all_videos if is_incident_video(v)]
        video_list = pos[:args.max_videos]

    print("\n" + "#" * 80)
    print(" AI CCTV FORENSIC SEARCH: DEEP STAGE INSPECTION SUITE")
    print(f" Inspecting {len(video_list)} Snatch Video(s) | Export Directory: {args.output_dir}")
    print("#" * 80)

    for v in video_list:
        inspect_video(
            evaluator,
            spec,
            v,
            max_frames=args.max_frames,
            export_video=args.export_video,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

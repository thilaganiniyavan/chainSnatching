"""Pattern Timeline & Keyframe Qualification Visualizer Tool.

Tracks when specific Behaviour Patterns (such as APPROACH_PATTERN,
INTERACTION_PATTERN, ESCAPE_PATTERN) get qualified in a video, prints the exact
frame numbers/timestamps and kinematics, and saves annotated visualization
keyframes and clips.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure root directory is in python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from src.detection.detector import Detector
from src.evaluation.research_live_evaluator import LiveResearchEvaluator, full_spec
from src.core.models.frame_context import FrameContext
from src.pipeline.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Pattern Qualification Frame Visualizer")
    parser.add_argument(
        "--video",
        type=str,
        default="Snatch 1.0/Chain Snatching Videos/Snatch Theft/1.mp4",
        help="Path to video file to inspect",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="ALL",
        help="Pattern name to track (e.g. ALL, APPROACH_PATTERN, INTERACTION_PATTERN, ESCAPE_PATTERN)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to process (default: 500)",
    )
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        default=True,
        help="Save annotated image when the pattern is qualified (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory where output keyframe images will be stored",
    )
    args = parser.parse_args()

    vpath = os.path.abspath(args.video)
    if not os.path.exists(vpath):
        print(f"Error: Video file not found at {vpath}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(vpath)

    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    evaluator = LiveResearchEvaluator(max_frames=args.max_frames)
    spec = full_spec("PatternVisualizer")
    stages, handles = evaluator._build_stages(spec, fps=fps, video_name=video_name)
    pipeline = Pipeline(stages=stages)
    detector = evaluator._ensure_detector()

    target_pattern = args.pattern.upper()

    print("\n" + "=" * 80)
    print(f" PATTERN QUALIFICATION TIMELINE INSPECTOR")
    print(f" Video     : {video_name} ({total_frames} total frames @ {fps:.1f} fps)")
    print(f" Target    : {target_pattern}")
    print("=" * 80 + "\n")

    frame_num = 0
    qualified_events = []
    seen_pattern_nodes = set()

    while cap.isOpened() and frame_num < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        dets = detector.detect(frame)
        context = FrameContext(
            frame=frame,
            frame_number=frame_num,
            timestamp=round(frame_num / fps, 3),
            detections=dets,
        )
        pipeline.run(context)

        # Inspect graphs for pattern nodes
        graphs = context.metadata.get("behaviour_graphs", [])
        for g in graphs:
            for node in g.nodes:
                matches_target = (target_pattern == "ALL") or (node.pattern_type == target_pattern)
                node_key = f"{g.interaction_id}-{node.pattern_id}-{node.pattern_type}"
                if matches_target and node_key not in seen_pattern_nodes:
                    seen_pattern_nodes.add(node_key)

                    # Extract current kinematics
                    tracks = context.tracks
                    track_map = {t.tracking_id: t for t in tracks}
                    p_track = track_map.get(g.person_track_id)
                    v_track = track_map.get(g.vehicle_track_id)

                    curr_dist = 0.0
                    if p_track and v_track and p_track.center and v_track.center:
                        curr_dist = float(np.linalg.norm(np.array(p_track.center) - np.array(v_track.center)))

                    event_info = {
                        "frame": frame_num,
                        "timestamp": round(frame_num / fps, 2),
                        "interaction_id": g.interaction_id,
                        "person_id": g.person_track_id,
                        "vehicle_id": g.vehicle_track_id,
                        "pattern_type": node.pattern_type,
                        "confidence": node.confidence,
                        "distance_px": round(curr_dist, 1),
                        "primitives": node.supporting_primitives,
                        "pattern_id": node.pattern_id,
                    }
                    qualified_events.append(event_info)

                    # Annotate frame for visualization
                    vis_frame = frame.copy()
                    h_img, w_img = vis_frame.shape[:2]

                    # Draw person and vehicle centers, bounding boxes and connecting line
                    if p_track and getattr(p_track, "detections", None):
                        last_det = p_track.detections[-1]
                        if hasattr(last_det, "bbox") and last_det.bbox:
                            bx, by, bw, bh = [int(v) for v in last_det.bbox]
                            cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    if v_track and getattr(v_track, "detections", None):
                        last_det = v_track.detections[-1]
                        if hasattr(last_det, "bbox") and last_det.bbox:
                            vx, vy, vw, vh = [int(v) for v in last_det.bbox]
                            cv2.rectangle(vis_frame, (vx, vy), (vx + vw, vy + vh), (0, 0, 255), 2)

                    if p_track and v_track and p_track.center and v_track.center:
                        p1 = (int(p_track.center[0]), int(p_track.center[1]))
                        p2 = (int(v_track.center[0]), int(v_track.center[1]))
                        cv2.line(vis_frame, p1, p2, (0, 165, 255), 2)
                        cv2.circle(vis_frame, p1, 6, (0, 255, 0), -1)
                        cv2.circle(vis_frame, p2, 6, (0, 0, 255), -1)
                        cv2.putText(vis_frame, f"Person #{g.person_track_id}", (p1[0] - 30, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"Vehicle #{g.vehicle_track_id}", (p2[0] - 30, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    # Overlay banner
                    if "APPROACH" in node.pattern_type:
                        banner_color = (0, 140, 255)
                    elif "INTERACTION" in node.pattern_type:
                        banner_color = (0, 215, 255)
                    elif "ESCAPE" in node.pattern_type:
                        banner_color = (0, 0, 255)
                    else:
                        banner_color = (0, 200, 0)

                    cv2.rectangle(vis_frame, (10, 10), (w_img - 10, 80), (20, 20, 20), -1)
                    cv2.rectangle(vis_frame, (10, 10), (w_img - 10, 80), banner_color, 2)
                    
                    title = f"PATTERN QUALIFIED: {node.pattern_type}"
                    details = f"Frame: {frame_num} | Time: {frame_num/fps:.2f}s | Dist: {curr_dist:.1f}px | Conf: {node.confidence:.2f} | Inter: {g.interaction_id}"
                    cv2.putText(vis_frame, title, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)
                    cv2.putText(vis_frame, details, (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    if args.save_keyframes:
                        save_name = f"{Path(video_name).stem}_{node.pattern_type}_frame_{frame_num:04d}.jpg"
                        out_path = os.path.join(args.output_dir, save_name)
                        cv2.imwrite(out_path, vis_frame)
                        event_info["saved_image"] = out_path

    cap.release()

    if qualified_events:
        print(f"✓ Found {len(qualified_events)} pattern qualification point(s):\n")
        print(f"{'Event #':<8} | {'Frame':<6} | {'Timestamp':<10} | {'Pattern Type':<22} | {'Distance':<10} | {'Confidence':<10} | Keyframe Image")
        print("-" * 105)
        for i, ev in enumerate(qualified_events, start=1):
            img_str = os.path.basename(ev.get("saved_image", "N/A"))
            print(f"{i:<8} | {ev['frame']:<6} | {ev['timestamp']:>5.2f}s     | {ev['pattern_type']:<22} | {ev['distance_px']:>6.1f} px  | {ev['confidence']:>6.2f}     | {img_str}")
        print("-" * 105)
        print(f"\nSaved keyframe images to: {os.path.abspath(args.output_dir)}\n")
    else:
        print(f"✗ No matching pattern was qualified within the first {frame_num} frames of {video_name}.\n")


if __name__ == "__main__":
    main()

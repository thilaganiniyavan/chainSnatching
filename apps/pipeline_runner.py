import cv2
import argparse
import sys
import os
import time

# Add project root to python path so it can be run from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def main():
    parser = argparse.ArgumentParser(description="AI Surveillance Pipeline Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or webcam ID")
    parser.add_argument("--backend", type=str, default="mediapipe", help="Pose estimation backend (mediapipe, rtmpose, vitpose, mmpose, openpose)")
    parser.add_argument("--norm", type=str, default="hip_centered", help="Skeleton normalization strategy (hip_centered, bbox, root_joint, image)")
    parser.add_argument("--action-backend", type=str, default="stgcn", help="Action recognition backend (stgcn, ctrgcn, msg3d, posec3d)")
    parser.add_argument("--fusion-strategy", type=str, default="weighted_confidence", help="Evidence fusion strategy (weighted_confidence, bayesian, rule_based, voting_based, weighted_averaging)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save periodic frames")
    args = parser.parse_args()

    # Initialize components
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    detector = Detector()
    
    tracking_stage = TrackingStage()
    relationship_stage = RelationshipStage(distance_threshold=150.0)
    interaction_stage = InteractionStage(distance_threshold=150.0)

    # Handle input
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
        
    if not cap.isOpened():
        print(f"Error opening video source: {args.input}")
        return

    # Setup output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle NaN or 0
        fps = 20.0
        
    video_basename = os.path.basename(args.input) if not args.input.isdigit() else f"webcam_{args.input}"

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/behaviour_graphs", exist_ok=True)
    os.makedirs("outputs/roi_clips", exist_ok=True)
    os.makedirs("outputs/pose_previews", exist_ok=True)
    os.makedirs("outputs/sequence_previews", exist_ok=True)
    os.makedirs("outputs/action_previews", exist_ok=True)
    os.makedirs("outputs/fusion_previews", exist_ok=True)
    os.makedirs("outputs/signature_previews", exist_ok=True)
    os.makedirs("outputs/forensic_thumbnails", exist_ok=True)
    os.makedirs("outputs/forensic_clips", exist_ok=True)

    # Initialize stages with video FPS and log paths
    behaviour_stage = BehaviourStage(
        fps=fps,
        output_log_path=os.path.join("outputs", "behaviour_log.json"),
    )
    reasoning_stage = ReasoningStage(
        fps=fps,
        output_json_path=os.path.join("outputs", "behaviour_events.json"),
        output_csv_path=os.path.join("outputs", "behaviour_events.csv"),
    )
    graph_stage = GraphReasoningStage(
        fps=fps,
        output_json_path=os.path.join("outputs", "behaviour_graph.json"),
        output_patterns_csv_path=os.path.join("outputs", "behaviour_patterns.csv"),
        output_transition_csv_path=os.path.join("outputs", "transition_matrix.csv"),
        export_diagrams_dir=os.path.join("outputs", "behaviour_graphs"),
    )
    roi_stage = ROISelectionStage(
        fps=fps,
        output_json_path=os.path.join("outputs", "interaction_rois.json"),
        output_csv_path=os.path.join("outputs", "roi_statistics.csv"),
        output_report_path=os.path.join("outputs", "roi_quality_report.md"),
        export_clips_dir=os.path.join("outputs", "roi_clips"),
    )
    pose_stage = PoseEstimationStage(
        backend_name=args.backend,
        fps=fps,
        output_json_path=os.path.join("outputs", "pose_results.json"),
        output_csv_path=os.path.join("outputs", "pose_statistics.csv"),
        output_report_path=os.path.join("outputs", "pose_quality_report.md"),
        export_previews_dir=os.path.join("outputs", "pose_previews"),
    )
    sequence_stage = SkeletonSequenceStage(
        normalization_method=args.norm,
        fps=fps,
        output_json_path=os.path.join("outputs", "skeleton_sequences.json"),
        output_csv_path=os.path.join("outputs", "sequence_statistics.csv"),
        output_report_path=os.path.join("outputs", "sequence_quality_report.md"),
        export_previews_dir=os.path.join("outputs", "sequence_previews"),
    )
    action_stage = ActionRecognitionStage(
        backend_name=args.action_backend,
        fps=fps,
        output_json_path=os.path.join("outputs", "action_results.json"),
        output_csv_path=os.path.join("outputs", "action_statistics.csv"),
        output_report_path=os.path.join("outputs", "action_recognition_report.md"),
        export_previews_dir=os.path.join("outputs", "action_previews"),
    )
    fusion_stage = BehaviourFusionStage(
        fusion_strategy=args.fusion_strategy,
        fps=fps,
        output_json_path=os.path.join("outputs", "fused_interactions.json"),
        output_csv_path=os.path.join("outputs", "fusion_statistics.csv"),
        output_report_path=os.path.join("outputs", "fusion_report.md"),
        export_previews_dir=os.path.join("outputs", "fusion_previews"),
    )
    snatch_stage = SnatchSignatureStage(
        output_json_path=os.path.join("outputs", "snatch_signature_results.json"),
        output_csv_path=os.path.join("outputs", "signature_statistics.csv"),
        output_report_path=os.path.join("outputs", "signature_report.md"),
        export_previews_dir=os.path.join("outputs", "signature_previews"),
    )
    indexing_stage = ForensicIndexingStage(
        video_id=video_basename,
        location="Camera 1",
        output_json_path=os.path.join("outputs", "forensic_events.json"),
        output_csv_path=os.path.join("outputs", "forensic_index.csv"),
        output_report_path=os.path.join("outputs", "forensic_index_report.md"),
        export_thumbnails_dir=os.path.join("outputs", "forensic_thumbnails"),
        export_clips_dir=os.path.join("outputs", "forensic_clips"),
    )

    pipeline = Pipeline(stages=[
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
    ])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputs/final_pipeline_output.avi', fourcc, fps, (width, height))
    
    if args.debug:
        os.makedirs("outputs/debug", exist_ok=True)

    frame_number = 0

    print(f"Pipeline started with pose '{args.backend}', norm '{args.norm}', action '{args.action_backend}', fusion '{args.fusion_strategy}'. Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number += 1
        
        # 2. Motion Detection
        fg_mask = mog2.apply(frame)
        motion_pixels = cv2.countNonZero(fg_mask)
        
        if motion_pixels <= 5000:
            continue
            
        # 3. YOLO Detection
        detections = detector.detect(frame)
        if not detections:
            continue
            
        # 4-17. Complete 13-Stage Pipeline Chain: Tracking -> Relationship -> Interaction -> Behaviour -> Reasoning -> GraphReasoning -> ROISelection -> PoseEstimation -> SkeletonSequence -> ActionRecognition -> BehaviourFusion -> SnatchSignature -> ForensicIndexing
        context = FrameContext(
            frame=frame,
            frame_number=frame_number,
            timestamp=time.time(),
            detections=detections
        )
        
        context = pipeline.run(context)
        
        # Use forensic_frame (includes all forensic index HUD overlays on top)
        viz_frame = context.metadata.get(
            "forensic_frame",
            context.metadata.get(
                "signature_frame",
                context.metadata.get(
                    "fusion_frame",
                    context.metadata.get(
                        "action_frame",
                        context.metadata.get(
                            "sequence_frame",
                            context.metadata.get(
                                "pose_frame",
                                context.metadata.get(
                                    "roi_frame",
                                    context.metadata.get(
                                        "graph_frame",
                                        context.metadata.get(
                                            "reasoning_frame",
                                            context.metadata.get(
                                                "behaviour_frame",
                                                context.metadata.get(
                                                    "relationship_frame",
                                                    context.metadata.get("trajectory_frame", frame.copy())
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        
        for track in context.tracks:
            bbox = track.metadata.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cx, cy = track.center
                x1, y1, x2, y2 = cx - 20, cy - 40, cx + 20, cy + 40
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                
            text_y = max(10, y1 - 10)
            
            cv2.putText(viz_frame, f"ID:{track.tracking_id} {track.class_name}", 
                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if track.average_speed is not None and track.direction is not None:
                info_text = f"Spd:{track.instantaneous_speed:.1f} Avg:{track.average_speed:.1f} Dir:{track.direction:.0f}"
                cv2.putText(viz_frame, info_text, 
                            (x1, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            
        out.write(viz_frame)
        
        # DEBUG MODE: Sample outputs every 100 frames
        if args.debug and frame_number % 100 == 0:
            cv2.imwrite("outputs/debug/motion_frame.jpg", fg_mask)
            
            det_frame = frame.copy()
            for det in detections:
                cv2.rectangle(det_frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            cv2.imwrite("outputs/debug/detection_frame.jpg", det_frame)
            
            cv2.imwrite("outputs/debug/trajectory_frame.jpg", context.metadata.get("trajectory_frame", frame))
            cv2.imwrite("outputs/debug/tracking_frame.jpg", viz_frame)
            if "behaviour_frame" in context.metadata:
                cv2.imwrite("outputs/debug/behaviour_frame.jpg", context.metadata["behaviour_frame"])
            if "reasoning_frame" in context.metadata:
                cv2.imwrite("outputs/debug/reasoning_frame.jpg", context.metadata["reasoning_frame"])
            if "graph_frame" in context.metadata:
                cv2.imwrite("outputs/debug/graph_frame.jpg", context.metadata["graph_frame"])
            if "roi_frame" in context.metadata:
                cv2.imwrite("outputs/debug/roi_frame.jpg", context.metadata["roi_frame"])
            if "pose_frame" in context.metadata:
                cv2.imwrite("outputs/debug/pose_frame.jpg", context.metadata["pose_frame"])
            if "sequence_frame" in context.metadata:
                cv2.imwrite("outputs/debug/sequence_frame.jpg", context.metadata["sequence_frame"])
            if "action_frame" in context.metadata:
                cv2.imwrite("outputs/debug/action_frame.jpg", context.metadata["action_frame"])
            if "fusion_frame" in context.metadata:
                cv2.imwrite("outputs/debug/fusion_frame.jpg", context.metadata["fusion_frame"])
            if "signature_frame" in context.metadata:
                cv2.imwrite("outputs/debug/signature_frame.jpg", context.metadata["signature_frame"])
            if "forensic_frame" in context.metadata:
                cv2.imwrite("outputs/debug/forensic_frame.jpg", context.metadata["forensic_frame"])
            
    # Finalize log outputs
    behaviour_stage.finalize()
    reasoning_stage.finalize()
    graph_stage.finalize()
    roi_stage.finalize()
    pose_stage.finalize()
    sequence_stage.finalize()
    action_stage.finalize()
    fusion_stage.finalize()
    snatch_stage.finalize()
    indexing_stage.finalize()

    cap.release()
    out.release()
    print("Pipeline execution completed. Output saved to outputs/final_pipeline_output.avi")
    print("Behaviour log saved to outputs/behaviour_log.json")
    print("Behaviour events saved to outputs/behaviour_events.json and outputs/behaviour_events.csv")
    print("Behaviour graph saved to outputs/behaviour_graph.json, outputs/behaviour_patterns.csv, outputs/transition_matrix.csv")
    print("Interaction ROIs saved to outputs/interaction_rois.json, outputs/roi_statistics.csv, outputs/roi_quality_report.md")
    print("Pose results saved to outputs/pose_results.json, outputs/pose_statistics.csv, outputs/pose_quality_report.md")
    print("Skeleton sequences saved to outputs/skeleton_sequences.json, outputs/sequence_statistics.csv, outputs/sequence_quality_report.md")
    print("Action results saved to outputs/action_results.json, outputs/action_statistics.csv, outputs/action_recognition_report.md")
    print("Fused interactions saved to outputs/fused_interactions.json, outputs/fusion_statistics.csv, outputs/fusion_report.md")
    print("Snatch signature results saved to outputs/snatch_signature_results.json, outputs/signature_statistics.csv, outputs/signature_report.md")
    print("Forensic events saved to outputs/forensic_events.json, outputs/forensic_index.csv, outputs/forensic_index_report.md")

if __name__ == "__main__":
    main()

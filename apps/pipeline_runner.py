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
from src.core.models.frame_context import FrameContext

def main():
    parser = argparse.ArgumentParser(description="AI Surveillance Pipeline Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or webcam ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save periodic frames")
    args = parser.parse_args()

    # Initialize components
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    detector = Detector()
    
    tracking_stage = TrackingStage()
    from src.pipeline.relationship_stage import RelationshipStage
    relationship_stage = RelationshipStage(distance_threshold=150.0)
    pipeline = Pipeline(stages=[tracking_stage, relationship_stage])

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
        
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputs/final_pipeline_output.avi', fourcc, fps, (width, height))
    
    if args.debug:
        os.makedirs("outputs/debug", exist_ok=True)

    frame_number = 0

    print("Pipeline started. Processing video in memory...")

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
            
        # 4, 5, 6. TrackingStage (Tracking, TrackHistory Update, MotionFeatureExtractor)
        context = FrameContext(
            frame=frame,
            frame_number=frame_number,
            timestamp=time.time(),
            detections=detections
        )
        
        context = pipeline.run(context)
        
        # 7. Draw on the frame (context.metadata["relationship_frame"] contains trajectories and relationships)
        viz_frame = context.metadata.get("relationship_frame", context.metadata.get("trajectory_frame", frame.copy()))
        
        for track in context.tracks:
            bbox = track.metadata.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                # Draw bounding box
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                # Fallback if no bbox is present
                cx, cy = track.center
                x1, y1, x2, y2 = cx - 20, cy - 40, cx + 20, cy + 40
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                
            text_y = max(10, y1 - 10)
            
            # Draw Track ID and class
            cv2.putText(viz_frame, f"ID:{track.tracking_id} {track.class_name}", 
                        (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw computed Motion Features
            if track.average_speed is not None and track.direction is not None:
                info_text = f"Spd:{track.instantaneous_speed:.1f} Avg:{track.average_speed:.1f} Dir:{track.direction:.0f}"
                cv2.putText(viz_frame, info_text, 
                            (x1, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            
        # 8. Write annotated frame to final output video
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
            
    cap.release()
    out.release()
    print("Pipeline execution completed. Output saved to outputs/final_pipeline_output.avi")

if __name__ == "__main__":
    main()

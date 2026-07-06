import cv2
import argparse
import sys
import os
import time

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.detector import Detector
from src.pipeline.pipeline import Pipeline
from src.pipeline.tracking_stage import TrackingStage
from src.pipeline.relationship_stage import RelationshipStage
from src.core.models.frame_context import FrameContext

def main():
    parser = argparse.ArgumentParser(description="Complete AI Surveillance Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Display the video while processing")
    parser.add_argument("--save", action="store_true", help="Save the output video")
    parser.add_argument("--debug", action="store_true", help="Print useful pipeline statistics")
    
    args = parser.parse_args()

    # Verify input exists (allow integers for webcams)
    is_webcam = args.input.isdigit()
    if not is_webcam and not os.path.exists(args.input):
        print(f"Error: Invalid input file. '{args.input}' does not exist.")
        sys.exit(1)

    try:
        input_source = int(args.input) if is_webcam else args.input
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{args.input}'.")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 20.0

        out = None
        if args.save:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Error: Cannot create output video '{args.output}'.")
                sys.exit(1)

        # Initialize existing modules
        mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        detector = Detector()
        
        tracking_stage = TrackingStage()
        relationship_stage = RelationshipStage(distance_threshold=150.0)
        
        pipeline = Pipeline(stages=[tracking_stage, relationship_stage])

        # Statistics
        stats_total_frames = 0
        stats_skipped_no_motion = 0
        stats_skipped_no_objects = 0
        stats_frames_processed = 0
        stats_total_persons = 0
        stats_total_vehicles = 0
        stats_max_tracks = 0
        
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            stats_total_frames += 1
            current_timestamp = stats_total_frames / fps
            
            # Stage 1: Motion Detection
            fg_mask = mog2.apply(frame)
            motion_pixels = cv2.countNonZero(fg_mask)
            
            if motion_pixels <= 5000:
                stats_skipped_no_motion += 1
                if args.show:
                    cv2.imshow("Surveillance Pipeline", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Stage 2: YOLO Detection
            detections = detector.detect(frame)
            if not detections:
                stats_skipped_no_objects += 1
                if args.show:
                    cv2.imshow("Surveillance Pipeline", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Accumulate detection stats
            for det in detections:
                if det.class_name == "person":
                    stats_total_persons += 1
                elif det.class_name in {"bicycle", "motorcycle", "car", "bus", "truck"}:
                    stats_total_vehicles += 1

            # Prepare FrameContext
            context = FrameContext(
                frame=frame,
                frame_number=stats_total_frames,
                timestamp=current_timestamp,
                detections=detections
            )

            # Stages 3-7: Execute Pipeline (Tracking, History, Trajectory, Motion Features, Relationships)
            context = pipeline.run(context)
            
            stats_frames_processed += 1
            active_tracks_count = len(context.tracks)
            if active_tracks_count > stats_max_tracks:
                stats_max_tracks = active_tracks_count

            # Visualization
            viz_frame = context.metadata.get("relationship_frame", context.metadata.get("trajectory_frame", frame.copy()))
            
            # Draw Bounding Boxes, Labels, and Motion Features
            for track in context.tracks:
                bbox = track.metadata.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # Draw bounding box
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    text_y = max(10, y1 - 10)
                    # Class Label and Tracking ID
                    class_id_text = f"ID {track.tracking_id} {track.class_name.capitalize()}"
                    cv2.putText(viz_frame, class_id_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Motion Features
                    features_text = []
                    if track.instantaneous_speed is not None:
                        features_text.append(f"Spd: {track.instantaneous_speed:.1f}")
                    if track.average_speed is not None:
                        features_text.append(f"Avg: {track.average_speed:.1f}")
                    if track.direction is not None:
                        features_text.append(f"Dir: {track.direction:.0f}")
                    if hasattr(track, 'distance_travelled') and track.distance_travelled is not None:
                        features_text.append(f"Dist: {track.distance_travelled:.1f}")
                        
                    if features_text:
                        features_str = " | ".join(features_text)
                        cv2.putText(viz_frame, features_str, (x1, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Draw Frame Information (Top-Left)
            cv2.putText(viz_frame, f"Frame Number: {stats_total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Timestamp: {current_timestamp:.2f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Objects Detected: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Tracks Active: {active_tracks_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if args.save and out is not None:
                out.write(viz_frame)

            if args.show:
                cv2.imshow("Surveillance Pipeline", viz_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Compute Final Statistics
        end_time = time.time()
        processing_time = end_time - start_time
        average_fps = stats_total_frames / processing_time if processing_time > 0 else 0

        # Print statistics to console
        print("="*50)
        print("PIPELINE EXECUTION STATISTICS")
        print("="*50)
        print(f"Total Frames                       : {stats_total_frames}")
        print(f"Frames Skipped (No Motion)         : {stats_skipped_no_motion}")
        print(f"Frames Skipped (No Relevant Objects): {stats_skipped_no_objects}")
        print(f"Frames Processed                   : {stats_frames_processed}")
        print(f"Average FPS                        : {average_fps:.2f}")
        print(f"Total Persons Detected             : {stats_total_persons}")
        print(f"Total Vehicles Detected            : {stats_total_vehicles}")
        print(f"Maximum Simultaneous Tracks        : {stats_max_tracks}")
        print(f"Processing Time                    : {processing_time:.2f}s")
        print("="*50)

        if args.debug:
            print("DEBUG: Execution completed gracefully.")

    except Exception as e:
        print(f"Unexpected pipeline exception: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

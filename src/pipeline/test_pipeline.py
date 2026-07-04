import cv2
import time
import sys
import os

# Add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.models.frame_context import FrameContext
from src.pipeline.pipeline import Pipeline
from src.pipeline.tracking_stage import TrackingStage

def main():
    # Initialize the stage and pipeline
    tracking_stage = TrackingStage()
    pipeline = Pipeline(stages=[tracking_stage])
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    frame_number = 0
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_number += 1
        current_time = time.time()
        
        # Create the context for the current frame
        context = FrameContext(
            frame=frame,
            frame_number=frame_number,
            timestamp=current_time
        )
        
        # Run the pipeline (this executes TrackingStage and TrajectoryVisualizer)
        context = pipeline.run(context)
        
        # Retrieve the visualization from metadata
        if "trajectory_frame" in context.metadata:
            viz_frame = context.metadata["trajectory_frame"]
            cv2.imshow("Trajectory Visualization", viz_frame)
        else:
            cv2.imshow("Trajectory Visualization", frame)

        # Print computed motion features to the terminal
        if context.tracks:
            print(f"--- Frame {frame_number} ---")
            for track in context.tracks:
                if track.average_speed is not None:
                    print(f"Track {track.tracking_id} [{track.class_name}]: "
                          f"Dist: {track.distance_travelled:.1f}px, "
                          f"Speed: {track.instantaneous_speed:.1f}px/f, "
                          f"Dir: {track.direction}")
            print("-" * 20)

        # Break on ESC
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

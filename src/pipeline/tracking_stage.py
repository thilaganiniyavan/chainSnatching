"""Tracking stage for the frame processing pipeline.

This stage uses the existing Tracker implementation to produce Track objects
and stores them on the shared FrameContext.
"""

from src.core.interfaces import Stage
from src.core.models import FrameContext, Track
from src.tracking.tracker import Tracker
from src.tracking.track_history_manager import TrackHistoryManager
from src.tracking.trajectory_visualizer import TrajectoryVisualizer
from src.tracking.motion_feature_extractor import MotionFeatureExtractor


class TrackingStage(Stage):
    """Pipeline stage that runs object tracking on the current frame."""

    def __init__(self) -> None:
        """Create a tracking stage with a single reusable tracker instance."""

        self.tracker = Tracker()
        self.history_manager = TrackHistoryManager()
        self.visualizer = TrajectoryVisualizer()
        self.feature_extractor = MotionFeatureExtractor()

    def process(self, context: FrameContext) -> FrameContext:
        """Run tracking on context.frame and store Track objects on the context."""

        results = self.tracker.track(context.frame)
        result = results[0]
        names = result.names
        tracks: list[Track] = []

        for box in result.boxes:
            tracking_id = int(box.id[0]) if box.id is not None else -1
            class_id = int(box.cls[0])
            class_name = names[class_id]

            # Compute center from bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            track = Track(
                tracking_id=tracking_id,
                class_name=class_name,
                detections=[],
                trajectory=[],
                first_seen=context.timestamp,
                last_seen=context.timestamp,
                metadata={"bbox": (x1, y1, x2, y2)},
                center=(cx, cy)
            )

            # Update history manager
            self.history_manager.update(track, context.timestamp)
            
            # Attach the history to the track object
            track.history = self.history_manager.get(tracking_id)

            # Compute motion features (modifies track in place)
            self.feature_extractor.compute(track)

            tracks.append(track)

        context.tracks = tracks
        
        # Visualize trajectories and store in metadata
        context.metadata["trajectory_frame"] = self.visualizer.draw(context.frame, context.tracks)
        
        return context
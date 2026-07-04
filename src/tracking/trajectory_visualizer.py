import cv2
import numpy as np
from src.core.models.track import Track

class TrajectoryVisualizer:
    """Visualizes the trajectory of tracked objects."""

    def draw(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """
        Draws trajectories on a copy of the frame.
        
        Args:
            frame: The image frame to draw on.
            tracks: A list of Track objects containing history.
            
        Returns:
            The annotated frame with trajectories.
        """
        # Create a copy to prevent overwriting the original frame
        viz_frame = frame.copy()
        
        color = (0, 255, 0)  # Consistent green color for now
        thickness = 2
        radius = 3

        for track in tracks:
            if track.history is None or not track.history.positions:
                continue

            positions = track.history.positions

            # Draw circles at each recorded position
            for pos in positions:
                cv2.circle(viz_frame, pos, radius, color, -1)

            # Draw lines between consecutive positions
            for i in range(1, len(positions)):
                pt1 = positions[i - 1]
                pt2 = positions[i]
                cv2.line(viz_frame, pt1, pt2, color, thickness)

        return viz_frame

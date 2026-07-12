import cv2
import numpy as np
from src.motion.base import MotionDetector

class KNNDetector(MotionDetector):
    """
    Detects motion using OpenCV's KNN (K-Nearest Neighbors) Background Subtractor.
    """

    def __init__(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = True,
        pixel_threshold: int = 5000
    ) -> None:
        """
        Initialize the detector.
        
        Args:
            history: Length of the history.
            dist2_threshold: Threshold on the squared distance between the pixel and the sample.
            detect_shadows: If True, the algorithm will detect and mark shadows.
            pixel_threshold: Number of foreground pixels required to trigger motion detection.
        """
        self.pixel_threshold = pixel_threshold
        self.subtractor = cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=dist2_threshold,
            detectShadows=detect_shadows
        )

    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Apply the KNN background subtractor and count foreground pixels.
        """
        fg_mask = self.subtractor.apply(frame)
        
        # Count non-zero pixels (foreground pixels)
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_detected = motion_pixels > self.pixel_threshold
        
        return motion_detected, fg_mask

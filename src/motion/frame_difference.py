import cv2
import numpy as np
from src.motion.base import MotionDetector

class FrameDifferenceDetector(MotionDetector):
    """
    Detects motion by calculating the absolute difference between the
    current frame and the previous frame.
    """

    def __init__(
        self, 
        threshold: int = 25, 
        pixel_threshold: int = 5000,
        blur_kernel_size: int = 21
    ) -> None:
        """
        Initialize the detector.
        
        Args:
            threshold: Intensity difference threshold to consider a pixel as changed.
            pixel_threshold: Number of changed pixels required to trigger motion detection.
            blur_kernel_size: Gaussian blur kernel size (must be positive and odd).
        """
        self.threshold = threshold
        self.pixel_threshold = pixel_threshold
        self.blur_kernel_size = (blur_kernel_size, blur_kernel_size)
        self.prev_gray = None

    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Compute absolute difference with previous frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            h, w = frame.shape[:2]
            return False, np.zeros((h, w), dtype=np.uint8)

        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        
        # Threshold the delta image
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Count motion pixels
        motion_pixels = cv2.countNonZero(thresh)
        motion_detected = motion_pixels > self.pixel_threshold

        # Update previous frame
        self.prev_gray = gray

        return motion_detected, thresh

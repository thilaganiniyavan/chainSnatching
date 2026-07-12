import cv2
import numpy as np
import warnings
from src.motion.base import MotionDetector

class GMMDetector(MotionDetector):
    """
    Detects motion using Gaussian Mixture Model (GMM) background subtraction.
    Attempts to use legacy MOG (cv2.bgsegm.createBackgroundSubtractorMOG) if available,
    falling back to MOG2.
    """

    def __init__(
        self,
        history: int = 500,
        nmixtures: int = 5,
        background_ratio: float = 0.7,
        noise_sigma: float = 0.0,
        pixel_threshold: int = 5000
    ) -> None:
        """
        Initialize the detector.
        
        Args:
            history: Length of the history.
            nmixtures: Number of Gaussian mixtures.
            background_ratio: Background ratio threshold.
            noise_sigma: Noise strength.
            pixel_threshold: Number of foreground pixels required to trigger motion detection.
        """
        self.pixel_threshold = pixel_threshold
        try:
            # Try to use legacy MOG subtractor from bgsegm
            self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
                history=history,
                nmixtures=nmixtures,
                backgroundRatio=background_ratio,
                noiseSigma=noise_sigma
            )
            self.using_legacy_mog = True
        except AttributeError:
            # Fallback to MOG2 (which is also GMM based)
            warnings.warn(
                "cv2.bgsegm.createBackgroundSubtractorMOG is not available in this OpenCV build. "
                "Falling back to MOG2 background subtractor (improved GMM)."
            )
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=16,
                detectShadows=True
            )
            self.using_legacy_mog = False

    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Apply the GMM background subtractor and count foreground pixels.
        """
        fg_mask = self.subtractor.apply(frame)
        
        # Count non-zero pixels (foreground pixels)
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_detected = motion_pixels > self.pixel_threshold
        
        return motion_detected, fg_mask

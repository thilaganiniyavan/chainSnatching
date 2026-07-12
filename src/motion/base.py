from abc import ABC, abstractmethod
import numpy as np

class MotionDetector(ABC):
    """Abstract base class for all motion detection algorithms."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Process the frame to detect motion.
        
        Args:
            frame: OpenCV BGR frame (np.ndarray)
            
        Returns:
            motion_detected: Boolean indicating if significant motion was detected.
            motion_mask: Binary mask (np.ndarray) of the same spatial dimensions,
                         where non-zero pixels represent motion.
        """
        pass


class NoFilteringDetector(MotionDetector):
    """
    Baseline motion detector that performs no filtering.
    Every frame is treated as containing motion.
    """

    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Returns True and a mask filled with 255 (all motion)."""
        h, w = frame.shape[:2]
        # Create a single channel mask with all 255 (indicating motion everywhere)
        motion_mask = np.ones((h, w), dtype=np.uint8) * 255
        return True, motion_mask

"""Realtime webcam entrypoint for the surveillance pipeline.

This application captures frames from the default webcam, processes them
through the pipeline, and displays the annotated output until ESC is pressed.
"""

import time

import cv2

from src.pipeline.detection_stage import DetectionStage
from src.pipeline.pipeline import Pipeline
from src.core.models import FrameContext


def main() -> None:
    """Run realtime detection on frames from the default webcam."""

    capture = cv2.VideoCapture(0)
    pipeline = Pipeline([DetectionStage()])
    frame_number = 0

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            context = FrameContext(
                frame=frame,
                frame_number=frame_number,
                timestamp=time.time(),
            )

            context = pipeline.run(context)
            annotated_frame = context.metadata["annotated_frame"]

            cv2.imshow("Realtime Detection", annotated_frame)

            frame_number += 1

            if cv2.waitKey(1) == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
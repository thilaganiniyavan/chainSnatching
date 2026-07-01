from ultralytics import YOLO
from configs.detection_config import (
    CONFIDENCE_THRESHOLD,
    ALLOWED_CLASSES
)
from src.detection.models import Detection


class Detector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.25
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence

    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        result = results[0]

        names = result.names

        for box in result.boxes:

            cls = int(box.cls[0])

            class_name = names[cls]

            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if class_name not in ALLOWED_CLASSES:
               continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            width = x2 - x1

            height = y2 - y1

            center_x = x1 + width // 2

            center_y = y1 + height // 2

            area = width * height

            detection = Detection(

                class_id=cls,

                class_name=class_name,

                confidence=confidence,

                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,

                center_x=center_x,
                center_y=center_y,

                width=width,
                height=height,

                area=area
            )

            detections.append(
                detection
            )

        return detections

    def draw(self, frame):

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        annotated = results[0].plot()

        detections = self.detect(frame)

        return annotated, detections
from ultralytics import YOLO


class Tracker:

    def __init__(self, model_path="yolo11n.pt"):

        self.model = YOLO(model_path)

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )

        return results
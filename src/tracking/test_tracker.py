import cv2

from src.tracking.tracker import Tracker

tracker = Tracker()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = tracker.track(frame)

    annotated = results[0].plot()

    cv2.imshow(
        "Tracking Engine",
        annotated
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()

cv2.destroyAllWindows()
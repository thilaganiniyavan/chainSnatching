import sys
import os
import cv2

# Add the project root to the python path so it can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.detection.detector import Detector

detector = Detector()
cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'outputs/relevant_motion_detection.avi',
    fourcc,
    20.0,
    (640, 480)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    fg_mask = mog2.apply(frame)

    motion_pixels = cv2.countNonZero(fg_mask)

    if motion_pixels <= 5000:
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground", fg_mask)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        continue

    annotated_frame, detections = detector.draw(frame)

    if not detections:
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground", fg_mask)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        continue

    out.write(annotated_frame)

    cv2.imshow("Original", annotated_frame)
    cv2.imshow("Foreground", fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
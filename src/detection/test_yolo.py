from ultralytics import YOLO
import cv2

# Load YOLOv11 Nano model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press ESC to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv11 Detection", annotated_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
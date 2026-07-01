from ultralytics import YOLO
import cv2


def main():

    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open webcam")

    print("Press ESC to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow(
            "YOLO + ByteTrack",
            annotated
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
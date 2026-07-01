import cv2

from src.detection.detector import Detector


def main():

    detector = Detector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open webcam")

    print("Press ESC to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame, detections = detector.draw(frame)

        for detection in detections:

            print(
                f"{detection.class_name:15}"
                f"{detection.confidence:.2f}"
            )

        print("-" * 50)

        cv2.imshow(
            "Detection Module",
            annotated_frame
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
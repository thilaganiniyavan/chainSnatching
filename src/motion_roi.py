import cv2
import numpy as np

cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    enhanced = cv2.convertScaleAbs(
        frame,
        alpha=2.0,
        beta=40
    )

    fg_mask = mog2.apply(enhanced)

    _, thresh = cv2.threshold(
        fg_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    display_frame = enhanced.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Area: {int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        display_frame,
        f"Brightness: {brightness:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Motion Mask", thresh)
    cv2.imshow("Motion ROI", display_frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
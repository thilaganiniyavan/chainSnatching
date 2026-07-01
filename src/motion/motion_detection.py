import cv2

cap = cv2.VideoCapture(0)

mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'outputs/motion_output.avi',
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

    if motion_pixels > 5000:
        out.write(frame)

        cv2.putText(
            frame,
            "MOTION DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Original", frame)
    cv2.imshow("Foreground", fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
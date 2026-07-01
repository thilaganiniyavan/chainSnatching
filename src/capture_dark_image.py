import cv2
import os

os.makedirs(
    "datasets/test_images",
    exist_ok=True
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam")

print("Press SPACE to capture image")
print("Press ESC to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow(
        "Capture Dark Image",
        frame
    )

    key = cv2.waitKey(1)

    if key == 32:  # SPACE

        output_path = "datasets/test_images/dark.jpg"

        cv2.imwrite(
            output_path,
            frame
        )

        print(
            f"Image saved to: {output_path}"
        )

        break

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
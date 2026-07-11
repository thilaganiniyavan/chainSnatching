import cv2

from src.depth.depth_estimator import DepthEstimator

model = DepthEstimator()

img = cv2.imread("datasets/test_images/test.avif")

if img is None:
    raise FileNotFoundError("Could not load image.")

depth, viz = model.estimate(img)

print(f"Depth map shape: {depth.shape}")
print(f"Depth range: {depth.min():.2f} - {depth.max():.2f}")

cv2.imshow("Original Image", img)
cv2.imshow("Depth Map", viz)

cv2.waitKey(0)
cv2.destroyAllWindows()
import os
import sys
import cv2
import torch

ZERO_DCE_PATH = "models/Zero-DCE/Zero-DCE_code"

sys.path.append(ZERO_DCE_PATH)

import model

device = torch.device("cpu")

# Load model
DCE_net = model.enhance_net_nopool().to(device)

DCE_net.load_state_dict(
    torch.load(
        f"{ZERO_DCE_PATH}/snapshots/Epoch99.pth",
        map_location=device
    )
)

DCE_net.eval()

# Load image
image_path = "datasets/test_images/dark.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

# Convert to RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Normalize
image_tensor = torch.from_numpy(
    image_rgb
).float() / 255.0

image_tensor = image_tensor.permute(
    2,
    0,
    1
)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)

# Enhance
with torch.no_grad():

    _, enhanced_image, _ = DCE_net(
        image_tensor
    )

# Convert back to image
enhanced_image = enhanced_image.squeeze(0)

enhanced_image = enhanced_image.permute(
    1,
    2,
    0
)

enhanced_image = enhanced_image.cpu().numpy()

enhanced_image = (
    enhanced_image * 255
).clip(
    0,
    255
).astype(
    "uint8"
)

enhanced_image = cv2.cvtColor(
    enhanced_image,
    cv2.COLOR_RGB2BGR
)

# Save result
os.makedirs(
    "outputs/enhanced",
    exist_ok=True
)

output_path = "outputs/enhanced/enhanced.jpg"

cv2.imwrite(
    output_path,
    enhanced_image
)

print(f"Saved: {output_path}")

# Show comparison
cv2.imshow("Original", image)
cv2.imshow("Enhanced", enhanced_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
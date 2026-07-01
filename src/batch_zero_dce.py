import os
import sys
import cv2
import torch

ZERO_DCE_PATH = "models/Zero-DCE/Zero-DCE_code"

sys.path.append(ZERO_DCE_PATH)

import model

device = torch.device("cpu")

DCE_net = model.enhance_net_nopool().to(device)

DCE_net.load_state_dict(
    torch.load(
        f"{ZERO_DCE_PATH}/snapshots/Epoch99.pth",
        map_location=device
    )
)

DCE_net.eval()

INPUT_DIR = "datasets/sample_lowlight"
OUTPUT_DIR = "outputs/enhanced_batch"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

valid_ext = (
    ".jpg",
    ".jpeg",
    ".png"
)

for filename in os.listdir(INPUT_DIR):

    if not filename.lower().endswith(valid_ext):
        continue

    image_path = os.path.join(
        INPUT_DIR,
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:
        print(f"Skipping {filename}")
        continue

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_tensor = torch.from_numpy(
        image_rgb
    ).float() / 255.0

    image_tensor = image_tensor.permute(
        2,
        0,
        1
    )

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        _, enhanced, _ = DCE_net(
            image_tensor
        )

    enhanced = enhanced.squeeze(0)

    enhanced = enhanced.permute(
        1,
        2,
        0
    )

    enhanced = enhanced.cpu().numpy()

    enhanced = (
        enhanced * 255
    ).clip(
        0,
        255
    ).astype(
        "uint8"
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2BGR
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        enhanced
    )

    print(
        f"Processed: {filename}"
    )

print("Done.")
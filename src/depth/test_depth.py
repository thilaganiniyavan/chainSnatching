import sys
from pathlib import Path

# Add Depth-Anything-V2 to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "Depth-Anything-V2"))

import torch
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }
}

model = DepthAnythingV2(**model_configs["vits"])

model.load_state_dict(
    torch.load(
        ROOT / "weights" / "depth_anything_v2_vits.pth",
        map_location="cpu"
    )
)

model.eval()

print("DepthAnything loaded successfully!")
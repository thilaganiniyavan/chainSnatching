"""
Detection configuration.

Only objects relevant to chain-snatching
are passed to the downstream modules.
"""

# Minimum confidence required
CONFIDENCE_THRESHOLD = 0.35


# COCO classes relevant to this project
ALLOWED_CLASSES = {

    "person",

    "bicycle",

    "motorcycle",

    "car",

    "bus",

    "truck"

}
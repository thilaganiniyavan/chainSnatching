"""Contact sheet of evenly spaced frames from a video (quick visual check of a clip).

Seeks to each frame, so it works on long recordings without loading them into memory.

    python scripts/contact_sheet.py VIDEO OUT.jpg [--n 8] [--start S] [--end S]
"""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
end = min(args.end or duration, duration) - 0.2
tiles = []
for t in np.linspace(args.start, end, args.n):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, im = cap.read()
    if not ok:
        continue
    im = cv2.resize(im, (400, 225))
    cv2.putText(im, f"{t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(im)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
cv2.imwrite(args.out, np.vstack(rows))

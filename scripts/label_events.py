"""Mark when the snatch happens in each clip (fills event_start_s / event_end_s in the labels CSV).

    python scripts/label_events.py                       # all snatch_bike clips without times
    python scripts/label_events.py --category normal     # any category

Keys:  space pause/play   a/d  step -/+ 0.2 s   j/l  jump -/+ 2 s
       s  mark start (first contact)   e  mark end (bike gone)
       w  save and next   x  skip   q  quit (saves what is done)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels" / "snatch1.0.csv"
DATA = ROOT / "data" / "samples" / "snatch1.0" / "Chain Snatching Videos"


def label(path: Path) -> tuple[float | None, float | None, bool]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t, playing, start, end = 0.0, True, None, None
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            t, playing = max(0.0, total - 0.1), False
            continue
        h, w = img.shape[:2]
        if w > 1280:
            img = cv2.resize(img, (1280, int(h * 1280 / w)))
        txt = f"{path.name}  t={t:6.2f}/{total:.1f}s  start={start}  end={end}  [space s e w x q]"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("label", img)
        key = cv2.waitKey(int(1000 / fps) if playing else 50) & 0xFF
        if playing:
            t = min(total, t + 1.0 / fps)
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            t = max(0.0, t - 0.2)
        elif key == ord("d"):
            t = min(total, t + 0.2)
        elif key == ord("j"):
            t = max(0.0, t - 2.0)
        elif key == ord("l"):
            t = min(total, t + 2.0)
        elif key == ord("s"):
            start = round(t, 2)
        elif key == ord("e"):
            end = round(t, 2)
        elif key == ord("w"):
            return start, end, False
        elif key == ord("x"):
            return None, None, False
        elif key == ord("q"):
            return start, end, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--category", default="snatch_bike")
    ap.add_argument("--all", action="store_true", help="also re-label clips that already have times")
    args = ap.parse_args()
    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    for row in rows:
        if row["category"] != args.category or (row["event_start_s"] and not args.all):
            continue
        start, end, quit_ = label(DATA / row["video"])
        if start is not None:
            row["event_start_s"], row["event_end_s"] = start, end if end is not None else ""
            with open(args.labels, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"{row['video']}: {start} - {end}")
        if quit_:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

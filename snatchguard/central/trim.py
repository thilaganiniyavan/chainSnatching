"""Trimmed detected videos: the moment of a detection cut as a proper, browser-playable MP4 (H.264).

Uploaded footage is cut from the original file (its own frame rate and quality); the browser camera is encoded from
every frame the server received, not only the frames the detector analysed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PRE_S = 10.0     # seconds kept before the detected contact
POST_S = 10.0    # and after it
_ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-an", "-movflags", "+faststart"]


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def trim_file(src: Path, start: float, end: float, out: Path) -> Path:
    start = max(0.0, start)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
                    "-t", f"{max(0.5, end - start):.3f}", *_ENCODE, str(out)], check=True, timeout=180)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("trimmed video is empty")
    return out


def frames_to_mp4(frames: list[tuple[float, bytes]], out: Path) -> Path:
    """frames: (wall time, JPEG bytes) in order; played back at the rate they were captured."""
    if len(frames) < 2:
        raise ValueError("not enough frames")
    fps = max(1.0, min(30.0, (len(frames) - 1) / max(0.1, frames[-1][0] - frames[0][0])))
    proc = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:.3f}",
                           "-c:v", "mjpeg", "-i", "-", *_ENCODE, str(out)],
                          input=b"".join(jpg for _, jpg in frames), check=True, timeout=180)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("trimmed video is empty")
    return out


# ---- boxes on the trimmed detected video --------------------------------------------------------------------------
DETECTED = (40, 200, 255)     # BGR: amber, the people the detector flagged
SNATCH = (40, 40, 230)        # red, the contact area around the moment of the grab
OTHER = (170, 170, 170)


def _boxes_at(frames: list[dict], key: str, t: float, max_gap: float = 1.0) -> dict[int, tuple[str, list[float]]]:
    """Track boxes at time t, interpolated between the analysed frames around it."""
    import bisect
    times = [f[key] for f in frames]
    i = bisect.bisect_left(times, t)
    before = frames[i - 1] if i > 0 else None
    after = frames[i] if i < len(frames) else None
    if before is None and after is None:
        return {}
    if before is None or after is None:
        f = before or after
        return {b[0]: (b[1], b[2:]) for b in f["boxes"]} if abs(f[key] - t) <= max_gap / 2 else {}
    if after[key] - before[key] > max_gap:
        f = before if t - before[key] <= after[key] - t else after
        return {b[0]: (b[1], b[2:]) for b in f["boxes"]} if abs(f[key] - t) <= max_gap / 2 else {}
    a = (t - before[key]) / max(1e-6, after[key] - before[key])
    nxt = {b[0]: b for b in after["boxes"]}
    out = {}
    for b in before["boxes"]:
        if b[0] in nxt:
            c = nxt[b[0]]
            out[b[0]] = (b[1], [p + (q - p) * a for p, q in zip(b[2:], c[2:])])
        elif a < 0.5:
            out[b[0]] = (b[1], b[2:])
    return out


def annotate(src: Path, times: list[float], data: dict, contact: float, out: Path, key: str = "t") -> Path:
    """Draw the flagged people (DETECTED) on every frame of `src`, and the contact area (SNATCH) within a second
    of the contact. times[i] is frame i's time on the same clock as data["frames"][*][key] and `contact`."""
    import cv2
    import numpy as np

    frames = sorted((f for f in data.get("frames", []) if f.get(key) is not None), key=lambda f: f[key])
    if not frames:
        raise ValueError("no boxes for this clip")
    flagged = {int(k) for k, v in data.get("roles", {}).items() if v}
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = subprocess.Popen([ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-framerate", f"{fps:.3f}", "-i", "-", *_ENCODE, str(out)],
                            stdin=subprocess.PIPE)
    thick = max(2, round(h / 240))
    font = max(0.5, h / 900)
    try:
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = times[min(i, len(times) - 1)] if times else i / fps
            boxes = _boxes_at(frames, key, t)
            union = None
            for tid, (cls, (x1, y1, x2, y2)) in boxes.items():
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
                if tid in flagged:
                    cv2.rectangle(img, p1, p2, DETECTED, thick)
                    cv2.putText(img, "DETECTED", (p1[0], max(int(18 * font), p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                font, DETECTED, thick)
                    union = [min(union[0], x1), min(union[1], y1), max(union[2], x2), max(union[3], y2)] if union \
                        else [x1, y1, x2, y2]
                elif cls == "person":
                    cv2.rectangle(img, p1, p2, OTHER, 1)
            if union and abs(t - contact) <= 1.0:
                pad = 0.02
                p1 = (int(max(0, union[0] - pad) * w), int(max(0, union[1] - pad) * h))
                p2 = (int(min(1, union[2] + pad) * w), int(min(1, union[3] + pad) * h))
                cv2.rectangle(img, p1, p2, SNATCH, thick + 1)
                cv2.putText(img, "SNATCH", (p1[0], min(h - 8, p2[1] + int(26 * font))), cv2.FONT_HERSHEY_SIMPLEX,
                            font * 1.1, SNATCH, thick + 1)
            proc.stdin.write(img.tobytes())
            i += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait(timeout=180)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("annotated video failed")
    return out

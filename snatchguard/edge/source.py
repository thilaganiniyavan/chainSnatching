"""Video sources that behave like a live camera.

`LiveSource` reads in a background thread and only ever hands out the newest
frame, so a slow processing loop drops frames instead of falling behind real
time, exactly like an edge box on a real RTSP camera. Video files are played
at their native speed (`realtime=True`) to simulate a camera from a laptop.

Reconnects with backoff when a stream drops. `status` feeds the heartbeat.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


@dataclass
class Frame:
    image: np.ndarray
    t: float        # seconds on the camera clock (video position for files, wall clock for streams)
    wall: float     # wall-clock time the frame was read
    seq: int
    segment: int = 0  # playlist item index (a new item = a different scene)


class LiveSource:
    def __init__(self, uri: str | int, *, realtime: bool = True, loop: bool = False,
                 transform: Callable[[np.ndarray], np.ndarray] | None = None, max_width: int = 1280) -> None:
        self.uri = int(uri) if isinstance(uri, str) and uri.isdigit() else uri
        self.is_file = isinstance(self.uri, str) and os.path.isfile(self.uri)
        self.realtime = realtime
        self.loop = loop
        self.transform = transform
        self.max_width = max_width

        self.status = "starting"        # starting | online | reconnecting | ended
        self.fps_in = 0.0
        self.src_fps = 0.0
        self._latest: Frame | None = None
        self._taken_seq = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"source-{uri}", daemon=True)

    # ------------------------------------------------------------------
    def start(self) -> "LiveSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Newest unseen frame, or None if nothing new arrived within `timeout`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                f = self._latest
                if f is not None and f.seq != self._taken_seq:
                    self._taken_seq = f.seq
                    # transform only the frames that are actually analysed, not every source frame
                    if self.transform is not None:
                        f = Frame(self.transform(self._resize(f.image)), f.t, f.wall, f.seq, f.segment)
                    return f
            if self.status == "ended":
                return None
            time.sleep(0.005)
        return None

    @property
    def ended(self) -> bool:
        return self.status == "ended"

    # ------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.uri) if isinstance(self.uri, int) else cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if 1.0 <= fps <= 120.0 else 25.0
        return cap

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _run(self) -> None:
        backoff = 1.0
        seq = 0
        loop_offset = 0.0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if self.is_file and not self.loop and self.status != "starting":
                    self.status = "ended"
                    return
                self.status = "reconnecting"
                log.warning("cannot open %s, retrying in %.0fs", self.uri, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            self.status = "online"
            backoff = 1.0
            start_wall = time.monotonic()
            window_start, window_count = time.monotonic(), 0
            last_pos = 0.0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                now = time.time()
                if self.is_file:
                    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if pos <= 0:
                        pos = last_pos + 1.0 / self.src_fps
                    last_pos = pos
                    t = loop_offset + pos
                    if self.realtime:
                        delay = (start_wall + pos) - time.monotonic()
                        if delay > 0:
                            self._stop.wait(delay)
                else:
                    t = now
                frame = Frame(self._resize(img), t, now, seq)
                seq += 1
                with self._lock:
                    self._latest = frame
                window_count += 1
                if time.monotonic() - window_start >= 2.0:
                    self.fps_in = window_count / (time.monotonic() - window_start)
                    window_start, window_count = time.monotonic(), 0
            cap.release()

            if self.is_file:
                if self.loop:
                    loop_offset += last_pos + 1.0
                    continue
                self.status = "ended"
                return
            self.status = "reconnecting"
            log.warning("stream %s dropped, reconnecting", self.uri)
            self._stop.wait(backoff)


# ---------------------------------------------------------------------------------------------------
# Playlists: several recordings (or parts of them) played back to back as one camera, for demos.
# ---------------------------------------------------------------------------------------------------

def resolve_playlist(spec: str, roots: list[Path]) -> list[tuple[str, float, float | None]]:
    """`playlist:<name>` reads configs/playlists/<name>.txt; `playlist:a.mp4|b.mp4` lists files inline.
    File lines: `path [start_s] [end_s]`; paths are looked up as given, then under each root."""
    body = spec.split(":", 1)[1]
    if "|" in body or Path(body).suffix:
        lines = body.split("|")
    else:
        lines = (roots[0] / "configs" / "playlists" / f"{body}.txt").read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.rsplit(None, 2)
        nums = []
        while len(parts) > 1 and _is_number(parts[-1]):
            nums.insert(0, float(parts.pop()))
        path = " ".join(parts) if len(parts) > 1 else parts[0]
        found = next((str(c) for c in [Path(path)] + [r / path for r in roots] if Path(c).exists()), None)
        if found is None:
            log.warning("playlist item not found, skipped: %s", path)
            continue
        out.append((found, nums[0] if nums else 0.0, nums[1] if len(nums) > 1 else None))
    if not out:
        raise FileNotFoundError(f"no playable items in {spec}")
    return out


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


class PlaylistSource(LiveSource):
    """Plays the items one after another (in real time) on one continuous camera clock."""

    def __init__(self, items: list[tuple[str, float, float | None]], *, realtime: bool = True, loop: bool = True,
                 transform=None, max_width: int = 1280) -> None:
        super().__init__("playlist", realtime=realtime, loop=loop, transform=transform, max_width=max_width)
        self.items = items
        self.is_file = True

    def _run(self) -> None:
        seq, segment, offset = 0, 0, 0.0
        self.status = "online"
        while not self._stop.is_set():
            for path, start, end in self.items:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    log.warning("cannot open %s, skipping", path)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.src_fps = fps if 1.0 <= fps <= 120.0 else 25.0
                if start:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                start_wall = time.monotonic()
                n = 0
                while not self._stop.is_set():
                    ok, img = cap.read()
                    if not ok:
                        break
                    rel = n / self.src_fps
                    n += 1
                    if end is not None and start + rel > end:
                        break
                    if self.realtime:
                        delay = start_wall + rel - time.monotonic()
                        if delay > 0:
                            self._stop.wait(delay)
                    frame = Frame(self._resize(img), offset + rel, time.time(), seq, segment)
                    seq += 1
                    with self._lock:
                        self._latest = frame
                    self.fps_in = self.src_fps
                cap.release()
                offset += n / self.src_fps + 1.0
                segment += 1
                if self._stop.is_set():
                    return
            if not self.loop:
                self.status = "ended"
                return

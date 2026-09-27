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
                if self.transform is not None:
                    img = self.transform(img)
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

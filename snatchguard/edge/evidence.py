"""Evidence capture: rolling frame buffer, annotated snapshot, WebM clip.

The last `pre_s + post_s` seconds of analysed frames are kept as JPEG bytes
(a few MB). When an alert fires, a snapshot goes out immediately with the
alert; the clip (from `pre_s` before the contact to `post_s` after the
decision) is written in a background thread and uploaded afterwards, so the
dashboard is notified without waiting for video encoding.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from snatchguard.core.types import Detection

log = logging.getLogger(__name__)

ROLE_COLORS = {"victim": (60, 200, 255), "vehicle": (40, 40, 230), "rider": (40, 120, 255)}


@dataclass
class _Buffered:
    t: float
    jpeg: bytes
    boxes: list[tuple[int, str, np.ndarray]]   # (track_id, class, box) in buffered-image pixels


@dataclass
class _ClipJob:
    event_id: str
    t0: float
    t1: float
    roles: dict[int, str]
    title: str


def draw_roles(img: np.ndarray, boxes: list[tuple[int, str, np.ndarray]], roles: dict[int, str]) -> np.ndarray:
    for tid, _cls, b in boxes:
        role = roles.get(tid)
        x1, y1, x2, y2 = (int(v) for v in b)
        if role is None:
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
            continue
        color = ROLE_COLORS["rider" if role.startswith("rider") else role]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, role.upper(), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def banner(img: np.ndarray, text: str, color=(20, 20, 200)) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), color, -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


class EvidenceRecorder:
    def __init__(self, out_dir: str | Path, pre_s: float = 8.0, post_s: float = 4.0, width: int = 960) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pre_s, self.post_s, self.width = pre_s, post_s, width
        self._buf: deque[_Buffered] = deque()
        self._pending: list[_ClipJob] = []
        self._jobs: queue.Queue[_ClipJob | None] = queue.Queue()
        self._done: queue.Queue[tuple[str, Path | None]] = queue.Queue()
        self._frames_for_job: dict[str, list[_Buffered]] = {}
        self._in_flight = 0
        self._thread = threading.Thread(target=self._writer, name="clip-writer", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def push(self, img: np.ndarray, t: float, detections: list[Detection]) -> None:
        h, w = img.shape[:2]
        scale = min(1.0, self.width / w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        boxes = [(d.track_id, d.cls.value, d.box * scale) for d in detections]
        self._buf.append(_Buffered(t, jpg.tobytes(), boxes))
        keep = self.pre_s + self.post_s + 6.0
        while self._buf and t - self._buf[0].t > keep:
            self._buf.popleft()
        # hand finished clip jobs (their end time has passed) to the writer thread
        for job in [j for j in self._pending if t >= j.t1]:
            self._pending.remove(job)
            self._frames_for_job[job.event_id] = [f for f in self._buf if job.t0 <= f.t <= job.t1]
            self._in_flight += 1
            self._jobs.put(job)

    def snapshot(self, event_id: str, at_t: float, roles: dict[int, str], title: str,
                 fallback: np.ndarray | None = None) -> Path:
        """Annotated still from the buffered frame closest to `at_t` (the moment of contact)."""
        best = min(self._buf, key=lambda f: abs(f.t - at_t), default=None)
        if best is not None:
            img = cv2.imdecode(np.frombuffer(best.jpeg, np.uint8), cv2.IMREAD_COLOR)
            draw_roles(img, best.boxes, roles)
        elif fallback is not None:
            img = fallback.copy()
        else:
            img = np.zeros((360, 640, 3), np.uint8)
        banner(img, title)
        path = self.dir / f"{event_id}_snapshot.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return path

    def schedule_clip(self, event_id: str, contact_t: float, decided_t: float, roles: dict[int, str], title: str) -> None:
        self._pending.append(_ClipJob(event_id, contact_t - self.pre_s, decided_t + self.post_s, roles, title))

    def finished(self) -> list[tuple[str, Path | None]]:
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
                self._in_flight -= 1
            except queue.Empty:
                return out

    def busy(self) -> bool:
        return bool(self._pending) or self._in_flight > 0

    def flush(self) -> None:
        """Write any pending clips with whatever frames exist (used on shutdown / end of file)."""
        for job in list(self._pending):
            self._pending.remove(job)
            self._frames_for_job[job.event_id] = [f for f in self._buf if f.t >= job.t0]
            self._in_flight += 1
            self._jobs.put(job)

    def close(self) -> None:
        self.flush()
        self._jobs.put(None)
        self._thread.join(timeout=30)

    # ------------------------------------------------------------------
    def _writer(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            frames = self._frames_for_job.pop(job.event_id, [])
            try:
                self._done.put((job.event_id, self._write_clip(job, frames)))
            except Exception:  # never let a codec problem kill the node
                log.exception("clip %s failed", job.event_id)
                self._done.put((job.event_id, None))

    def _write_clip(self, job: _ClipJob, frames: list[_Buffered]) -> Path | None:
        if len(frames) < 2:
            return None
        span = frames[-1].t - frames[0].t
        fps = max(2.0, min(30.0, (len(frames) - 1) / span)) if span > 0 else 8.0
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        path = self.dir / f"{job.event_id}_clip.webm"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        if not writer.isOpened():
            path = path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        for f in frames:
            img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
            draw_roles(img, f.boxes, job.roles)
            banner(img, f"{job.title}   t={f.t:.1f}s")
            writer.write(img)
        writer.release()
        return path

"""Isolated browser-camera sessions; one detector and bounded frame buffer per device."""
from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from snatchguard.central.footage import FootageRunner


@dataclass
class BrowserCamera:
    id: str
    token: str
    runner: FootageRunner
    created: float = field(default_factory=time.time)
    at: float = 0.0
    connected: bool = False
    closed: bool = False
    seq: int = 0
    jpg: bytes | None = None
    frames: deque = field(default_factory=deque)
    buffered_bytes: int = 0
    cv: threading.Condition = field(default_factory=threading.Condition)

    def push(self, data: bytes) -> None:
        with self.cv:
            self.at = time.time()
            self.jpg = data
            self.seq += 1
            self.frames.append((self.at, data))
            self.buffered_bytes += len(data)
            while self.frames and (self.frames[0][0] < self.at - 40 or self.buffered_bytes > 32 * 1024 * 1024):
                self.buffered_bytes -= len(self.frames.popleft()[1])
            self.cv.notify_all()

    def start(self, source: str, name: str, port: int, site: str):
        with self.cv:
            if self.closed or not self.connected or not self.jpg or time.time() - self.at > 15:
                raise RuntimeError("Connect this device's camera before starting detection.")
            return self.runner.start(source, name, port, live=True, site=site)

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.connected = False
            self.cv.notify_all()
        self.runner.stop()


class BrowserCameras:
    def __init__(self, directory: Path, key: str, limit: int = 4):
        self.directory, self.key, self.limit = directory, key, limit
        self.sessions: dict[str, BrowserCamera] = {}
        self.lock = threading.RLock()

    def create(self) -> BrowserCamera:
        with self.lock:
            self.expire()
            if len(self.sessions) >= 256:
                raise RuntimeError("Camera session limit reached. Wait before reconnecting.")
            if sum(not s.closed for s in self.sessions.values()) >= self.limit:
                raise RuntimeError("Camera capacity reached. Stop a camera before connecting another device.")
            sid = secrets.token_hex(12)
            session = BrowserCamera(sid, secrets.token_urlsafe(32), FootageRunner(self.directory / sid, self.key))
            self.sessions[sid] = session
            return session

    def owned(self, sid: str, token: str) -> BrowserCamera | None:
        with self.lock:
            s = self.sessions.get(sid)
            return s if s and secrets.compare_digest(s.token, token) else None

    def for_node(self, node: str | None) -> BrowserCamera | None:
        with self.lock:
            return next((s for s in self.sessions.values() if s.runner.job and s.runner.job['node_id'] == node), None)

    def expire(self) -> list[BrowserCamera]:
        expired = []
        with self.lock:
            now = time.time()
            for sid, s in list(self.sessions.items()):
                # A connected but frozen tab must not keep a detector or an "online" camera indefinitely.
                if not s.closed and now - (s.at or s.created) > (15 if s.connected else 60):
                    s.close()
                    expired.append(s)
                if s.closed and now - (s.at or s.created) > 60:
                    with s.cv:
                        s.frames.clear()
                        s.jpg = None
                        s.buffered_bytes = 0
                if s.closed and now - (s.at or s.created) > 3600:
                    del self.sessions[sid]
        return expired

    def close(self) -> None:
        with self.lock:
            for s in self.sessions.values():
                s.close()

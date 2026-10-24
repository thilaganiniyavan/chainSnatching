"""Live two-stage verification for one camera.

Every analysed frame goes into a short ring buffer and through stage 1. When a clip candidate's window has
passed (AFTER_S after its onset), its frames are cut from the buffer and handed to a worker thread that runs the
verifier, so the network never slows the per-frame loop. The worker takes the newest clip first: when it falls
behind, a fresh contact is still checked promptly, and clips that have waited longer than `max_lag_s` are dropped
and counted, so every alert still has its footage in the evidence buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from snatchguard.core.types import Detection
from snatchguard.verifier.candidates import CandidateStage, ClipCandidate
from snatchguard.verifier.model import AFTER_S, BEFORE_S, cut, sample_times

log = logging.getLogger(__name__)


@dataclass
class Verified:
    clip: ClipCandidate
    score: float
    decided_t: float


class _Runner(threading.Thread):
    def __init__(self, verifier, name: str, max_lag_s: float) -> None:
        super().__init__(name=name, daemon=True)
        self.verifier = verifier
        self.max_lag_s = max_lag_s
        self._cv = threading.Condition()
        self._todo: deque = deque()
        self._done: list[Verified] = []
        self._busy = False
        self._halt = False
        self.scored = 0
        self.dropped = 0
        self.error: str | None = None

    def offer(self, clip: ClipCandidate, frames: np.ndarray, t: float) -> None:
        with self._cv:
            self._todo.append((clip, frames, t, time.monotonic()))
            self._expire()
            self._cv.notify_all()

    def _expire(self) -> None:          # lock held
        now = time.monotonic()
        while self._todo and now - self._todo[0][3] > self.max_lag_s:
            self._todo.popleft()
            self.dropped += 1

    def drain(self) -> list[Verified]:
        with self._cv:
            out, self._done = self._done, []
        return out

    def wait_idle(self, timeout: float) -> None:
        with self._cv:
            self._cv.wait_for(lambda: self.error is not None or not (self._todo or self._busy), timeout)

    def stop(self) -> None:
        with self._cv:
            self._halt = True
            self._cv.notify_all()

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._halt:
                    self._expire()
                    if self._todo:
                        break
                    self._cv.wait(timeout=1.0)
                if self._halt:
                    return
                batch = [self._todo.pop()]      # newest first; on a CPU, batching does not raise throughput
                self._busy = True
            try:
                scores = self.verifier.score(np.stack([b[1] for b in batch]))
            except Exception as exc:          # the rules keep running without the verifier
                log.exception("%s: verifier failed and is switched off", self.name)
                with self._cv:
                    self.error = str(exc)[:200]
                    self._busy = False
                    self._cv.notify_all()
                return
            with self._cv:
                self._done += [Verified(c, float(s), t) for (c, _, t, _), s in zip(batch, scores)]
                self.scored += len(batch)
                self._busy = False
                self._cv.notify_all()


class VerifierStage:
    def __init__(self, verifier, cam_id: str, threshold: float | None = None, max_lag_s: float = 20.0) -> None:
        self.stage = CandidateStage()
        self.threshold = verifier.threshold if threshold is None else float(threshold)
        self.max_lag_s = float(max_lag_s)
        self._ring: deque[tuple[float, np.ndarray]] = deque()
        self._pending: list[ClipCandidate] = []
        self.candidates = 0
        self.runner = _Runner(verifier, f"verifier-{cam_id}", self.max_lag_s)
        self.runner.start()

    def step(self, t: float, img_bgr: np.ndarray, dets: list[Detection]) -> list[Verified]:
        """Feed one analysed frame (the image the detector saw, and its raw tracker detections).
        Returns the clips verified since the last call that scored at or above the threshold."""
        self._ring.append((t, img_bgr))
        while self._ring and t - self._ring[0][0] > BEFORE_S + AFTER_S + 1.0:
            self._ring.popleft()
        new = self.stage.step(t, dets)
        self.candidates += len(new)
        self._pending += new
        ready = [c for c in self._pending if t >= c.t + AFTER_S]
        self._pending = [c for c in self._pending if t < c.t + AFTER_S]
        for clip in ready:
            self.runner.offer(clip, self._frames(clip), t)
        return [v for v in self.runner.drain() if v.score >= self.threshold]

    def _frames(self, clip: ClipCandidate) -> np.ndarray:
        ts = np.array([x[0] for x in self._ring])
        out = [cut(self._ring[int(np.argmin(np.abs(ts - tt)))][1], clip.crop) for tt in sample_times(clip.t)]
        return np.stack(out)

    def flush(self) -> None:
        """Scene over: verify the clips still inside their window with the frames seen so far (past the last frame
        the last one repeats, as in training)."""
        if self._ring:
            t = self._ring[-1][0]
            for clip in self._pending:
                self.runner.offer(clip, self._frames(clip), t)
        self._pending = []

    def reset(self) -> None:
        """New scene (next playlist item): flush, then start over with new track ids and an empty buffer."""
        self.flush()
        self.stage = CandidateStage()
        self._ring.clear()

    def finish(self, timeout: float = 60.0) -> list[Verified]:
        """End of the stream: flush, wait for the network to finish, return the alerts still owed."""
        self.flush()
        self.runner.wait_idle(timeout)
        return [v for v in self.runner.drain() if v.score >= self.threshold]

    def stop(self) -> None:
        self.runner.stop()
        close = getattr(self.runner.verifier, "close", None)     # the verifier's own process, if any
        if close is not None:
            close()

    def status(self) -> dict:
        r = self.runner
        return {"verifier": "error" if r.error else "on",
                "verifier_stats": {"candidates": self.candidates, "scored": r.scored, "dropped": r.dropped}}

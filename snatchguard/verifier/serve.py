"""The verifier in its own process at the lowest CPU priority.

On a small edge CPU the video network and the detector compete for the same cores. On a crowded scene on a 2-core
laptop the rules analyse 2.7 frames/s alone, 1.3 with the network in the camera process, and 2.3 with it in a child
process at `nice 19`, which only gets the time detection leaves idle: the network falls behind instead (late clips
are dropped and counted by the live stage). Clips go over stdin as raw uint8, scores come back over stdout as
float32.

    python -m snatchguard.verifier.serve <model> <threads> <device> <nice>
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from snatchguard.verifier.model import NF, SIDE

_CLIP_BYTES = NF * SIDE * SIDE * 3


def _read(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


class ProcessVerifier:
    """Same interface as Verifier (`threshold`, `score`, `backend`), running in a niced child process."""

    def __init__(self, model_path: str | Path, threads: int = 2, device: str = "cpu", nice: int = 19) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "snatchguard.verifier.serve", str(model_path),
                                      str(int(threads)), device, str(int(nice))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        head = self.proc.stdout.readline().decode().split()
        if not head or head[0] != "ready":
            self.close()
            raise RuntimeError("verifier process failed to start: " + " ".join(head))
        self.threshold, self.backend = float(head[1]), head[2]

    def score(self, clips: np.ndarray) -> np.ndarray:
        clips = np.ascontiguousarray(clips, np.uint8)
        self.proc.stdin.write(struct.pack("<I", len(clips)) + clips.tobytes())
        self.proc.stdin.flush()
        return np.frombuffer(_read(self.proc.stdout, 4 * len(clips)), np.float32).copy()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main() -> None:
    model, threads, device, nice = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    if nice:
        os.nice(nice)                            # before any library starts its worker threads
    from snatchguard.verifier.model import Verifier
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                      # library chatter must not corrupt the score stream
    v = Verifier(model, threads, device)
    out.write(f"ready {v.threshold!r} {v.encoder.backend}\n".encode())
    out.flush()
    inp = sys.stdin.buffer
    while True:
        try:
            n = struct.unpack("<I", _read(inp, 4))[0]
            clips = np.frombuffer(_read(inp, n * _CLIP_BYTES), np.uint8).reshape(n, NF, SIDE, SIDE, 3)
        except EOFError:
            return
        out.write(v.score(clips).astype(np.float32).tobytes())
        out.flush()


if __name__ == "__main__":
    main()

"""Video sources: playlist resolution and playback (segments, continuous camera clock)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from snatchguard.edge.source import PlaylistSource, resolve_playlist


def _video(path: Path, frames: int, fps: float = 10.0, shade: int = 0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(frames):
        w.write(np.full((120, 160, 3), (shade + i * 3) % 255, np.uint8))
    w.release()
    return path


def test_resolve_playlist_file_and_inline(tmp_path):
    a, b = _video(tmp_path / "a clip.avi", 5), _video(tmp_path / "b.avi", 5)
    (tmp_path / "configs" / "playlists").mkdir(parents=True)
    (tmp_path / "configs" / "playlists" / "x.txt").write_text(
        f"# comment\n{a.name} 0.2 0.4\nmissing.avi\n{b.name}\n", encoding="utf-8")
    items = resolve_playlist("playlist:x", [tmp_path])
    assert [(Path(p).name, s, e) for p, s, e in items] == [("a clip.avi", 0.2, 0.4), ("b.avi", 0.0, None)]
    inline = resolve_playlist(f"playlist:{a}|{b} 0.1", [tmp_path])
    assert len(inline) == 2 and inline[1][1] == 0.1


def test_playlist_plays_items_in_order_on_one_clock(tmp_path):
    a, b = _video(tmp_path / "a.avi", 12), _video(tmp_path / "b.avi", 8, shade=100)
    src = PlaylistSource([(str(a), 0.0, None), (str(b), 0.0, None)], realtime=True, loop=False).start()
    seen = []
    deadline = time.time() + 15
    while time.time() < deadline:
        f = src.read(timeout=0.5)
        if f is None:
            if src.ended:
                break
            continue
        seen.append((f.segment, f.t))
    src.stop()
    segments = [s for s, _ in seen]
    assert segments[0] == 0 and segments[-1] == 1 and segments == sorted(segments)
    ts = [t for _, t in seen]
    assert ts == sorted(ts)                              # camera clock never goes backwards
    assert min(t for s, t in seen if s == 1) > max(t for s, t in seen if s == 0)

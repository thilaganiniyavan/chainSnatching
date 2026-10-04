"""Edge node supervisor.

* one process per camera (a crash or hang in one camera can't take the others down)
* restarts crashed camera processes with backoff (watchdog)
* sends a heartbeat with every camera's health + thumbnail to central
* runs the publisher that drains the store-and-forward outbox
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import signal
import time
from datetime import datetime, timezone

from snatchguard.core.config import REPO_ROOT, load_yaml
from snatchguard.edge.outbox import Outbox
from snatchguard.edge.publisher import Publisher

log = logging.getLogger(__name__)
VERSION = "0.1.0"


def _camera_process(cam: dict, node: dict, status_q, stop_evt, log_level: int) -> None:
    logging.basicConfig(level=log_level, format=f"%(asctime)s [{cam['id']}] %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # the supervisor coordinates shutdown
    from snatchguard.edge.camera import CameraWorker
    CameraWorker(cam, node, status_q).run(stop_evt)


class EdgeNode:
    def __init__(self, cfg: dict, log_level: int = logging.INFO) -> None:
        self.cfg = cfg
        self.log_level = log_level
        self.node_id = cfg.get("node_id", "edge")
        central = cfg.get("central", {})
        outbox_path = cfg.get("outbox", "runtime/edge/outbox.db")
        self.outbox = Outbox(outbox_path if os.path.isabs(outbox_path) else REPO_ROOT / outbox_path)
        self.publisher = Publisher(self.outbox, central.get("url", "http://127.0.0.1:8000"),
                                   central.get("api_key", ""))
        self.heartbeat_s = float(central.get("heartbeat_s", 10))
        self.ctx = mp.get_context("spawn")
        self.status_q = self.ctx.Queue()
        self.stop_evt = self.ctx.Event()
        self.procs: dict[str, mp.Process] = {}
        self.restarts: dict[str, int] = {}
        self.next_start: dict[str, float] = {}
        self.finished: set[str] = set()
        self.cam_status: dict[str, dict] = {}
        self.started = time.time()

    @classmethod
    def from_file(cls, path: str, **kw) -> "EdgeNode":
        return cls(load_yaml(path), **kw)

    # ------------------------------------------------------------------
    def _spawn(self, cam: dict) -> None:
        p = self.ctx.Process(target=_camera_process, name=f"cam-{cam['id']}", daemon=True,
                             args=(cam, self.cfg, self.status_q, self.stop_evt, self.log_level))
        p.start()
        self.procs[cam["id"]] = p
        log.info("camera %s -> pid %d", cam["id"], p.pid)

    def run(self, until_done: bool = False) -> None:
        stop_requested = False

        def _on_signal(*_):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        self.publisher.start()
        cams = {c["id"]: c for c in self.cfg.get("cameras", [])}
        for cam in cams.values():
            self._spawn(cam)

        last_hb = 0.0
        while not stop_requested:
            self._drain_status()
            now = time.monotonic()
            for cid, p in list(self.procs.items()):
                if p.is_alive():
                    continue
                st = self.cam_status.get(cid, {})
                if p.exitcode == 0 and st.get("status") in ("ended", "stopped"):
                    self.finished.add(cid)
                    del self.procs[cid]
                    log.info("camera %s finished its source", cid)
                    continue
                # crashed: restart with backoff
                n = self.restarts.get(cid, 0)
                if cid not in self.next_start:
                    delay = min(60.0, 2.0 ** n)
                    self.next_start[cid] = now + delay
                    log.error("camera %s died (exit %s), restart #%d in %.0fs", cid, p.exitcode, n + 1, delay)
                    self.cam_status[cid] = {**st, "id": cid, "status": "restarting"}
                elif now >= self.next_start.pop(cid):
                    self.restarts[cid] = n + 1
                    self._spawn(cams[cid])
            if now - last_hb >= self.heartbeat_s:
                last_hb = now
                self.publisher.heartbeat(self._heartbeat())
            if until_done and not self.procs and self.outbox.pending() == 0:
                break
            time.sleep(0.3)

        self.shutdown()

    def shutdown(self) -> None:
        log.info("shutting down")
        self.stop_evt.set()
        for p in self.procs.values():
            p.join(timeout=40)
            if p.is_alive():
                p.terminate()
        self._drain_status()
        deadline = time.monotonic() + 15
        while self.outbox.pending() and time.monotonic() < deadline:
            time.sleep(0.5)
        self.publisher.heartbeat(self._heartbeat())
        self.publisher.stop()
        if self.outbox.pending():
            log.warning("%d item(s) still queued; they will be sent on next start", self.outbox.pending())

    def _drain_status(self) -> None:
        while True:
            try:
                st = self.status_q.get_nowait()
            except queue.Empty:
                return
            self.cam_status[st["id"]] = st

    def _heartbeat(self) -> dict:
        cams = []
        for cam in self.cfg.get("cameras", []):
            st = dict(self.cam_status.get(cam["id"], {"id": cam["id"], "name": cam.get("name"), "status": "starting"}))
            st["restarts"] = self.restarts.get(cam["id"], 0)
            if cam["id"] in self.procs and not self.procs[cam["id"]].is_alive() and st.get("status") == "online":
                st["status"] = "restarting"
            cams.append(st)
        return {
            "node_id": self.node_id,
            "site": self.cfg.get("site"),
            "version": VERSION,
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "uptime_s": round(time.time() - self.started),
            "load_avg": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,   # not on Windows
            "outbox_pending": self.outbox.pending(),
            "cameras": cams,
        }

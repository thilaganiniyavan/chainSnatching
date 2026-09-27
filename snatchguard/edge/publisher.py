"""Delivers outbox items and heartbeats to the central server over HTTP(S).

Kinds:
  event  -> POST /api/v1/events            (multipart: meta JSON + snapshot)
  clip   -> POST /api/v1/events/{id}/clip  (multipart: clip file)
Heartbeats are best-effort (the next one replaces a lost one) and never queued.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from pathlib import Path

import httpx

from snatchguard.edge.outbox import Outbox, OutboxItem

log = logging.getLogger(__name__)


class Publisher(threading.Thread):
    def __init__(self, outbox: Outbox, base_url: str, api_key: str) -> None:
        super().__init__(name="publisher", daemon=True)
        self.outbox = outbox
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.client = httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
        self.connected = False
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            items = self.outbox.due()
            for item in items:
                if self._stop.is_set():
                    break
                self._deliver(item)
            self._stop.wait(0.5 if items else 1.0)

    def _deliver(self, item: OutboxItem) -> None:
        try:
            if item.kind == "event":
                resp = self._post_files("/api/v1/events", {"meta": json.dumps(item.payload)}, item.files)
            elif item.kind == "clip":
                resp = self._post_files(f"/api/v1/events/{item.payload['event_id']}/clip", {}, item.files)
            else:
                log.error("unknown outbox kind %s, dropping", item.kind)
                self.outbox.done(item.id)
                return
        except (httpx.HTTPError, OSError) as exc:
            self.connected = False
            self.outbox.retry_later(item.id, item.attempts, str(exc))
            log.warning("delivery of %s #%d failed (%s), will retry", item.kind, item.id, exc)
            return
        self.connected = True
        if resp.status_code < 300 or resp.status_code == 409:
            self.outbox.done(item.id)
            log.info("delivered %s #%d", item.kind, item.id)
        elif 400 <= resp.status_code < 500 and resp.status_code not in (401, 403, 404, 408, 429):
            log.error("central rejected %s #%d: %s %s (dropping)", item.kind, item.id, resp.status_code, resp.text[:200])
            self.outbox.done(item.id)
        else:
            self.outbox.retry_later(item.id, item.attempts, f"HTTP {resp.status_code}")

    def _post_files(self, path: str, data: dict, files: dict[str, str]) -> httpx.Response:
        handles = {}
        try:
            for field, fpath in files.items():
                p = Path(fpath)
                if p.exists():
                    handles[field] = (p.name, open(p, "rb"), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            return self.client.post(self.base + path, data=data, files=handles or None, headers=self.headers)
        finally:
            for _, fh, _ in handles.values():
                fh.close()

    def heartbeat(self, payload: dict) -> bool:
        try:
            resp = self.client.post(self.base + "/api/v1/heartbeat", json=payload, headers=self.headers)
            self.connected = resp.status_code < 300
        except (httpx.HTTPError, OSError):
            self.connected = False
        return self.connected

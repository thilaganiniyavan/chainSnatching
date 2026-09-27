"""Store-and-forward outbox (SQLite).

Every alert, clip upload and tamper report is written here first and only
deleted after the central server acknowledges it. If the network or the
server is down, items wait and are retried with exponential backoff; a power
cut loses nothing. Event ids are generated on the edge, so a retry after a
lost acknowledgement can't create a duplicate (central inserts idempotently).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutboxItem:
    id: int
    kind: str
    payload: dict
    files: dict[str, str]
    attempts: int


class Outbox:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                files TEXT NOT NULL,
                created REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_try REAL NOT NULL,
                last_error TEXT)""")
            self._db.commit()

    def put(self, kind: str, payload: dict, files: dict[str, str] | None = None) -> int:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO outbox (kind, payload, files, created, next_try) VALUES (?, ?, ?, ?, ?)",
                (kind, json.dumps(payload), json.dumps(files or {}), now, now))
            self._db.commit()
            return int(cur.lastrowid)

    def due(self, limit: int = 20) -> list[OutboxItem]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, kind, payload, files, attempts FROM outbox WHERE next_try <= ? ORDER BY id LIMIT ?",
                (time.time(), limit)).fetchall()
        return [OutboxItem(r[0], r[1], json.loads(r[2]), json.loads(r[3]), r[4]) for r in rows]

    def done(self, item_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE id = ?", (item_id,))
            self._db.commit()

    def retry_later(self, item_id: int, attempts: int, error: str) -> None:
        delay = min(60.0, 2.0 ** min(attempts, 6))
        with self._lock:
            self._db.execute("UPDATE outbox SET attempts = ?, next_try = ?, last_error = ? WHERE id = ?",
                             (attempts + 1, time.time() + delay, error[:500], item_id))
            self._db.commit()

    def pending(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()

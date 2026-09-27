"""Central storage (SQLite for the laptop demo; the schema ports directly to PostgreSQL)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,              -- snatch | tamper | camera_offline | camera_online
    level TEXT NOT NULL,             -- CONFIRMED | SUSPECT | TAMPER | TAMPER_CLEARED | OFFLINE | ONLINE
    node_id TEXT, site TEXT, camera_id TEXT, camera_name TEXT,
    occurred_at TEXT, detected_at TEXT, received_at TEXT NOT NULL,
    score REAL, confidence TEXT, lighting TEXT,
    payload TEXT NOT NULL,
    snapshot TEXT, clip TEXT,
    status TEXT NOT NULL DEFAULT 'new',   -- new | acknowledged | confirmed | false_alarm
    reviewed_at TEXT, review_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at DESC);
CREATE TABLE IF NOT EXISTS cameras (
    key TEXT PRIMARY KEY,            -- node_id/camera_id
    node_id TEXT, camera_id TEXT, name TEXT, site TEXT,
    status TEXT, last_seen TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY, site TEXT, last_seen TEXT, offline INTEGER DEFAULT 0, payload TEXT
);
"""

EVENT_COLS = ["id", "type", "level", "node_id", "site", "camera_id", "camera_name", "occurred_at", "detected_at",
              "received_at", "score", "confidence", "lighting", "payload", "snapshot", "clip", "status",
              "reviewed_at", "review_note"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
            self.db.commit()

    # ---- events ------------------------------------------------------------
    def insert_event(self, meta: dict[str, Any], snapshot: str | None) -> bool:
        """Idempotent insert; returns False if the event id already exists."""
        row = {
            "id": meta["event_id"], "type": meta.get("type", "snatch"), "level": meta.get("level", "SUSPECT"),
            "node_id": meta.get("node_id"), "site": meta.get("site"), "camera_id": meta.get("camera_id"),
            "camera_name": meta.get("camera_name"), "occurred_at": meta.get("occurred_at"),
            "detected_at": meta.get("detected_at"), "received_at": now_iso(), "score": meta.get("score"),
            "confidence": meta.get("confidence"), "lighting": meta.get("lighting"),
            "payload": json.dumps(meta), "snapshot": snapshot, "clip": None,
            "status": "new" if meta.get("level") in ("CONFIRMED", "SUSPECT", "TAMPER", "OFFLINE") else "acknowledged",
            "reviewed_at": None, "review_note": None,
        }
        with self.lock:
            cur = self.db.execute(
                f"INSERT OR IGNORE INTO events ({','.join(EVENT_COLS)}) VALUES ({','.join('?' * len(EVENT_COLS))})",
                [row[c] for c in EVENT_COLS])
            self.db.commit()
            return cur.rowcount == 1

    def set_clip(self, event_id: str, clip: str) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE events SET clip = ? WHERE id = ?", (clip, event_id))
            self.db.commit()
            return cur.rowcount == 1

    def review(self, event_id: str, status: str, note: str | None) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE events SET status = ?, reviewed_at = ?, review_note = ? WHERE id = ?",
                                  (status, now_iso(), note, event_id))
            self.db.commit()
            return cur.rowcount == 1

    def get_event(self, event_id: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._event(r) if r else None

    def list_events(self, limit: int = 100, type_: str | None = None, status: str | None = None,
                    level: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM events WHERE 1=1", []
        for col, val in (("type", type_), ("status", status), ("level", level)):
            if val:
                q += f" AND {col} = ?"
                args.append(val)
        q += " ORDER BY received_at DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        return [self._event(r) for r in rows]

    @staticmethod
    def _event(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        return d

    def stats(self, since_iso: str) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT type, level, status, COUNT(*) n FROM events WHERE received_at >= ? GROUP BY type, level, status",
                (since_iso,)).fetchall()
        out = {"confirmed": 0, "suspect": 0, "tamper": 0, "open": 0, "reviewed_true": 0, "reviewed_false": 0}
        for r in rows:
            if r["type"] == "snatch":
                out["confirmed" if r["level"] == "CONFIRMED" else "suspect"] += r["n"]
                if r["status"] == "confirmed":
                    out["reviewed_true"] += r["n"]
                elif r["status"] == "false_alarm":
                    out["reviewed_false"] += r["n"]
            if r["type"] == "tamper" and r["level"] == "TAMPER":
                out["tamper"] += r["n"]
            if r["status"] == "new":
                out["open"] += r["n"]
        return out

    # ---- health ------------------------------------------------------------
    def upsert_heartbeat(self, hb: dict) -> None:
        ts = now_iso()
        with self.lock:
            self.db.execute(
                "INSERT INTO nodes (node_id, site, last_seen, offline, payload) VALUES (?, ?, ?, 0, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET site=excluded.site, last_seen=excluded.last_seen, offline=0, "
                "payload=excluded.payload",
                (hb["node_id"], hb.get("site"), ts, json.dumps({k: v for k, v in hb.items() if k != "cameras"})))
            for cam in hb.get("cameras", []):
                cam = {k: v for k, v in cam.items() if k != "thumbnail"}
                self.db.execute(
                    "INSERT INTO cameras (key, node_id, camera_id, name, site, status, last_seen, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET name=excluded.name, "
                    "status=excluded.status, last_seen=excluded.last_seen, payload=excluded.payload",
                    (f"{hb['node_id']}/{cam['id']}", hb["node_id"], cam["id"], cam.get("name"), hb.get("site"),
                     cam.get("status"), ts, json.dumps(cam)))
            self.db.commit()

    def mark_node_offline(self, node_id: str) -> None:
        with self.lock:
            self.db.execute("UPDATE nodes SET offline = 1 WHERE node_id = ?", (node_id,))
            self.db.execute("UPDATE cameras SET status = 'offline' WHERE node_id = ?", (node_id,))
            self.db.commit()

    def nodes(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM nodes").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]

    def cameras(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM cameras ORDER BY node_id, camera_id").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]

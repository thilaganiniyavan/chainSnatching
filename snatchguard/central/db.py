"""Central storage (SQLite for the laptop demo; the schema ports directly to PostgreSQL)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS portal_settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
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
CREATE TABLE IF NOT EXISTS event_links (
    event_a TEXT NOT NULL REFERENCES events(id),
    event_b TEXT NOT NULL REFERENCES events(id),
    relation TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY(event_a, event_b)
);
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


# Alerts held for AI review, or rejected by it, are not shown anywhere (they stay in the table for audit).
BASE_VISIBLE = ("type NOT IN ('camera_offline', 'camera_online') AND "
                "COALESCE(json_extract(payload, '$.ai_review.state'), '') NOT IN ('pending', 'rejected', 'merged') "
                "AND received_at >= COALESCE((SELECT value FROM portal_settings WHERE key='activity_start'), '')")
DELETED = ("(COALESCE(json_extract(payload, '$.deleted'),0)=1 OR COALESCE(json_extract(payload, '$.dismissed'),0)=1 "
           "OR COALESCE(json_extract(payload, '$.clip_removed'),0)=1 "
           "OR INSTR(COALESCE(review_note,''),'[Deleted from portal]')>0 "
           "OR INSTR(COALESCE(review_note,''),'[Dismissed from portal]')>0 "
           "OR INSTR(COALESCE(review_note,''),'[Video removed from alert]')>0)")
VISIBLE = BASE_VISIBLE + " AND NOT " + DELETED



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

    def archive_activity(self) -> dict:
        """Start an empty operational view; keep prior events, media and camera rows recoverable."""
        cutoff = now_iso()
        with self.lock:
            events = self.db.execute("SELECT COUNT(*) FROM events WHERE received_at < ?", (cutoff,)).fetchone()[0]
            cameras = self.db.execute("SELECT COUNT(*) FROM cameras WHERE last_seen < ?", (cutoff,)).fetchone()[0]
            self.db.execute("INSERT INTO portal_settings VALUES ('activity_start', ?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (cutoff,))
            self.db.commit()
        return {"events": events, "cameras": cameras, "since": cutoff}

    def restore_activity(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM portal_settings WHERE key='activity_start'")
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
            cur = self.db.execute(
                "UPDATE events SET clip=CASE WHEN json_extract(payload, '$.clip_removed')=1 THEN NULL ELSE ? END, "
                "payload=CASE WHEN json_extract(payload, '$.clip_removed')=1 "
                "THEN json_set(payload, '$.removed_clip', ?) ELSE payload END WHERE id=?", (clip, clip, event_id))
            self.db.commit()
            return cur.rowcount == 1

    def delete_alert(self, event_id: str, restore: bool = False) -> bool:
        with self.lock:
            if restore:
                cur = self.db.execute(
                    "UPDATE events SET clip=COALESCE(clip,json_extract(payload,'$.removed_clip')), "
                    "payload=json_remove(payload,'$.deleted','$.dismissed','$.clip_removed','$.removed_clip'), "
                    "review_note=TRIM(REPLACE(REPLACE(REPLACE(COALESCE(review_note,''), "
                    "'[Deleted from portal]',''),'[Dismissed from portal]',''),'[Video removed from alert]','')) WHERE id=?",
                    (event_id,))
            else:
                cur = self.db.execute("UPDATE events SET payload=json_set(payload,'$.deleted',json('true')) WHERE id=?",
                                      (event_id,))
            self.db.commit()
            return cur.rowcount == 1

    def dismiss_tamper(self, event_id: str) -> bool:
        with self.lock:
            cur = self.db.execute(
                "UPDATE events SET payload=json_set(payload, '$.dismissed', json('true')) WHERE id=? AND type='tamper'",
                (event_id,))
            self.db.commit()
            return cur.rowcount == 1

    def video_visibility(self, event_id: str, visible: bool) -> bool:
        with self.lock:
            if visible:
                cur = self.db.execute(
                    "UPDATE events SET clip=CASE WHEN json_extract(payload, '$.clip_removed')=1 "
                    "THEN json_extract(payload, '$.removed_clip') ELSE clip END, "
                    "payload=json_remove(payload, '$.clip_removed', '$.removed_clip'), "
                    "review_note=REPLACE(COALESCE(review_note,''), '[Video removed from alert]', '') WHERE id=?", (event_id,))
            else:
                cur = self.db.execute(
                    "UPDATE events SET payload=json_set(payload, '$.clip_removed', json('true'), '$.removed_clip', clip), "
                    "clip=NULL WHERE id=? AND clip IS NOT NULL", (event_id,))
            self.db.commit()
            return cur.rowcount == 1

    def set_ai_review(self, event_id: str, review: dict) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE events SET payload = json_set(payload, '$.ai_review', json(?)) WHERE id = ?",
                                  (json.dumps(review), event_id))
            self.db.commit()
            return cur.rowcount == 1

    def set_level(self, event_id: str, level: str) -> None:
        with self.lock:
            self.db.execute("UPDATE events SET level = ?, payload = json_set(payload, '$.level', ?) WHERE id = ?",
                            (level, level, event_id))
            self.db.commit()

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
                    level: str | None = None, node_id: str | None = None, deleted: bool | None = False) -> list[dict]:
        visibility = BASE_VISIBLE if deleted is None else BASE_VISIBLE + (" AND " if deleted else " AND NOT ") + DELETED
        q, args = "SELECT * FROM events WHERE " + visibility, []
        for col, val in (("type", type_), ("status", status), ("level", level), ("node_id", node_id)):
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
        if "[Video removed from alert]" in (d.get("review_note") or ""):
            d["payload"]["clip_removed"] = True
            d["payload"]["removed_clip"] = d["clip"]
            d["clip"] = None
        return d

    def stats(self, since_iso: str, real_only: bool = False) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT type, level, status, COUNT(*) n FROM events WHERE received_at >= ? AND " + VISIBLE + " "
                + ("AND COALESCE(node_id, '') NOT IN ('edge-sim-01', 'recorded-examples') AND COALESCE(site, '') != 'Scripted demonstration' " if real_only else "")
                + "GROUP BY type, level, status",
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

    def link_events(self, first: str, second: str, relation: str, note: str) -> None:
        a, b = sorted((first, second))
        with self.lock:
            self.db.execute("INSERT INTO event_links VALUES (?, ?, ?, ?, ?) "
                            "ON CONFLICT(event_a, event_b) DO UPDATE SET relation=excluded.relation, note=excluded.note",
                            (a, b, relation, note, now_iso()))
            self.db.commit()

    def event_links(self, event_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM event_links WHERE event_a=? OR event_b=? ORDER BY created_at DESC",
                                   (event_id, event_id)).fetchall()
        return [{**dict(r), "other_id": r["event_b"] if r["event_a"] == event_id else r["event_a"]} for r in rows]

    def unlink_events(self, first: str, second: str) -> None:
        a, b = sorted((first, second))
        with self.lock:
            self.db.execute("DELETE FROM event_links WHERE event_a=? AND event_b=?", (a, b))
            self.db.commit()

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
            rows = self.db.execute("SELECT * FROM nodes WHERE last_seen >= COALESCE((SELECT value FROM portal_settings WHERE key='activity_start'), '')").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]

    def cameras(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM cameras WHERE last_seen >= COALESCE((SELECT value FROM portal_settings WHERE key='activity_start'), '') ORDER BY node_id, camera_id").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"] or "{}")} for r in rows]

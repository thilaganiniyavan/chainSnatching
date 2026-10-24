"""Phone alerts: each alert the central server receives is pushed to Telegram with its snapshot, and the evidence
clip follows as a reply when the edge uploads it.

Setup (about 2 minutes, one time):
    1. In Telegram, message @BotFather, send /newbot and copy the token it gives you.
    2. Send your new bot any message (e.g. "hi"); add the bot to a group instead to alert several people.
    3. python -m snatchguard.central.notify chat-id --token <token>      # prints the chat id
    4. python -m snatchguard.central.notify test --token <token> --chat <chat id>
    5. Start central with SNATCHGUARD_TELEGRAM_TOKEN and SNATCHGUARD_TELEGRAM_CHAT set.

Environment: SNATCHGUARD_TELEGRAM_TOKEN, SNATCHGUARD_TELEGRAM_CHAT (comma-separated for several chats),
SNATCHGUARD_NOTIFY_LEVELS (default CONFIRMED,SUSPECT,OFFLINE,TAMPER), SNATCHGUARD_PUBLIC_URL (dashboard address
put in each alert, e.g. http://192.168.1.20:8000).

Sending runs on its own thread with retries, so a slow or offline phone network never delays the API. Each
delivery is appended to <data>/notify.jsonl with the delays contact -> edge decision -> central -> phone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
LEVELS = ("CONFIRMED", "SUSPECT", "OFFLINE", "TAMPER")
TRIES = 4                 # attempts per message (waits 2, 4, 8 s between them)
MAX_VIDEO_BYTES = 49_000_000   # Telegram bots may send up to 50 MB
HEADLINE = {"CONFIRMED": "🚨 CHAIN SNATCHING", "SUSPECT": "⚠️ Possible snatching", "OFFLINE": "📵 Camera offline",
            "TAMPER": "🛑 Camera tampering"}


def _t(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def caption(ev: dict, public_url: str = "") -> str:
    level = ev.get("level", "")
    lines = [f"{HEADLINE.get(level, level)} - {ev.get('camera_name') or ev.get('camera_id') or ev.get('node_id')}"]
    if ev.get("site"):
        lines[0] += f" ({ev['site']})"
    occurred = _t(ev.get("occurred_at"))
    if occurred is not None:
        lines.append(datetime.fromtimestamp(occurred).strftime("%d %b %H:%M:%S"))
    if ev.get("type") == "snatch":
        bits = []
        if ev.get("mode") and ev["mode"] != "unknown":
            bits.append(ev["mode"].replace("_", " "))
        if ev.get("score") is not None:
            bits.append(f"score {float(ev['score']):.2f}")
        if ev.get("model_score") is not None:
            bits.append(f"model {float(ev['model_score']):.2f}")
        if bits:
            lines.append(" · ".join(bits))
        detected = _t(ev.get("detected_at"))
        if occurred is not None and detected is not None:
            lines.append(f"alert {detected - occurred:.1f} s after the contact")
        if level == "SUSPECT":
            lines.append("Check the clip before acting.")
    elif ev.get("detail") or ev.get("condition"):
        lines.append(str(ev.get("detail") or ev["condition"]))
    if ev.get("supersedes"):
        lines.append("(upgrade of an earlier alert)")
    if public_url:
        lines.append(f"Dashboard: {public_url.rstrip('/')}/")
    return "\n".join(lines)[:1024]


class TelegramNotifier:
    """`event` and `clip` only queue work; a daemon thread sends it."""

    def __init__(self, token: str, chats: list[str], levels=LEVELS, public_url: str = "",
                 log_path: Path | None = None, client: httpx.Client | None = None, retry_wait_s: float = 2.0):
        self.base = f"{API}/bot{token}"
        self.chats, self.levels, self.public_url = chats, set(levels), public_url
        self.log_path, self.retry_wait_s = log_path, retry_wait_s
        self.client = client or httpx.Client(timeout=30.0)
        self.sent: dict[tuple[str, str], int] = {}     # (event id, chat) -> Telegram message id, for clip replies
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="telegram", daemon=True)
        self.thread.start()

    @classmethod
    def from_env(cls, data_dir: Path) -> "TelegramNotifier | None":
        token, chats = os.environ.get("SNATCHGUARD_TELEGRAM_TOKEN"), os.environ.get("SNATCHGUARD_TELEGRAM_CHAT")
        if not token or not chats:
            return None
        levels = os.environ.get("SNATCHGUARD_NOTIFY_LEVELS", ",".join(LEVELS)).split(",")
        n = cls(token, [c.strip() for c in chats.split(",") if c.strip()], [lv.strip().upper() for lv in levels],
                os.environ.get("SNATCHGUARD_PUBLIC_URL", ""), data_dir / "notify.jsonl")
        log.info("phone alerts on: Telegram, %d chat(s), levels %s", len(n.chats), ",".join(sorted(n.levels)))
        return n

    def event(self, ev: dict, snapshot: Path | None = None) -> None:
        if ev.get("level") in self.levels:
            self.q.put(("event", ev, snapshot, time.time()))

    def clip(self, event_id: str, path: Path) -> None:
        self.q.put(("clip", event_id, path, time.time()))

    def close(self, timeout: float = 10.0) -> None:
        """Waits until the queue is sent (or `timeout`), then stops the thread."""
        self.q.put(None)
        self.thread.join(timeout)

    def _run(self) -> None:
        while (job := self.q.get()) is not None:
            try:
                if job[0] == "event":
                    self._send_event(*job[1:])
                else:
                    self._send_clip(*job[1:])
            except Exception:                              # never let one bad alert stop the thread
                log.exception("phone alert failed")

    def _call(self, method: str, data: dict, files: dict | None = None) -> dict | None:
        for attempt in range(TRIES):
            for f in (files or {}).values():
                f[1].seek(0)
            try:
                r = self.client.post(f"{self.base}/{method}", data=data, files=files)
                body = r.json()
                if body.get("ok"):
                    return body["result"]
                if r.status_code == 429:                   # rate limited: Telegram says how long to wait
                    time.sleep(float(body.get("parameters", {}).get("retry_after", self.retry_wait_s)))
                    continue
                if 400 <= r.status_code < 500:             # bad token / chat: retrying will not help
                    log.error("Telegram %s refused: %s", method, body.get("description"))
                    return None
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("Telegram %s attempt %d failed: %s", method, attempt + 1, exc)
            time.sleep(self.retry_wait_s * 2 ** attempt)
        return None

    def _send_event(self, ev: dict, snapshot: Path | None, queued: float) -> None:
        text = caption(ev, self.public_url)
        for chat in self.chats:
            reply = self.sent.get((ev["supersedes"], chat)) if ev.get("supersedes") else None
            data = {"chat_id": chat, **({"reply_to_message_id": reply} if reply else {})}
            if snapshot is not None and snapshot.exists():
                with open(snapshot, "rb") as f:
                    res = self._call("sendPhoto", {**data, "caption": text}, {"photo": ("snapshot.jpg", f, "image/jpeg")})
            else:
                res = self._call("sendMessage", {**data, "text": text})
            if res is not None:
                self.sent[(ev["event_id"], chat)] = res["message_id"]
            self._record(ev, chat, queued, res is not None)

    def _send_clip(self, event_id: str, path: Path, queued: float) -> None:
        if not path.exists() or path.stat().st_size > MAX_VIDEO_BYTES:
            return
        for chat in self.chats:
            msg = self.sent.get((event_id, chat))
            if msg is None:                                # this alert's level is not sent to phones
                continue
            with open(path, "rb") as f:
                self._call("sendVideo", {"chat_id": chat, "reply_to_message_id": msg, "caption": "evidence clip"},
                           {"video": (path.name, f, "video/mp4" if path.suffix == ".mp4" else "video/webm")})

    def _record(self, ev: dict, chat: str, queued: float, ok: bool) -> None:
        if self.log_path is None:
            return
        now = time.time()
        occurred, detected, received = (_t(ev.get(k)) for k in ("occurred_at", "detected_at", "received_at"))
        rec = {"event_id": ev["event_id"], "level": ev.get("level"), "chat": chat, "delivered": ok,
               "sent_at": datetime.now(timezone.utc).isoformat(),
               "contact_to_decision_s": None if None in (occurred, detected) else round(detected - occurred, 2),
               "decision_to_central_s": None if None in (detected, received) else round(received - detected, 2),
               "central_to_phone_s": round(now - queued, 2),
               "contact_to_phone_s": None if occurred is None else round(now - occurred, 2)}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Telegram phone alerts: find the chat id, send a test alert")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat-id", help="list the chats that have messaged the bot")
    c.add_argument("--token", default=os.environ.get("SNATCHGUARD_TELEGRAM_TOKEN"))
    t = sub.add_parser("test", help="send a test alert")
    t.add_argument("--token", default=os.environ.get("SNATCHGUARD_TELEGRAM_TOKEN"))
    t.add_argument("--chat", default=os.environ.get("SNATCHGUARD_TELEGRAM_CHAT"))
    t.add_argument("--snapshot", help="a JPEG to attach")
    a = ap.parse_args()
    if not a.token:
        ap.error("--token (or SNATCHGUARD_TELEGRAM_TOKEN) is required")
    if a.cmd == "chat-id":
        r = httpx.get(f"{API}/bot{a.token}/getUpdates", timeout=30.0).json()
        if not r.get("ok"):
            raise SystemExit(f"Telegram refused: {r.get('description')}")
        chats = {u[k]["chat"]["id"]: u[k]["chat"] for u in r["result"] for k in ("message", "my_chat_member")
                 if k in u}
        if not chats:
            raise SystemExit("No messages yet: send your bot a message in Telegram, then run this again.")
        for cid, ch in chats.items():
            print(cid, ch.get("title") or " ".join(filter(None, (ch.get("first_name"), ch.get("last_name")))))
        return
    if not a.chat:
        ap.error("--chat (or SNATCHGUARD_TELEGRAM_CHAT) is required")
    n = TelegramNotifier(a.token, [x.strip() for x in a.chat.split(",")], retry_wait_s=1.0)
    now = datetime.now(timezone.utc).isoformat()
    n.event({"event_id": "test", "type": "snatch", "level": "CONFIRMED", "camera_name": "Test camera",
             "site": "SnatchGuard test", "occurred_at": now, "detected_at": now, "mode": "on_bike", "score": 0.99},
            Path(a.snapshot) if a.snapshot else None)
    n.close(60.0)
    print("sent" if n.sent else "failed: see the log above")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

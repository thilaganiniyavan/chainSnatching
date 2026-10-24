"""Phone alerts: what is sent to Telegram, clip replies, retries, level filter, delay log, central wiring."""

from __future__ import annotations

import json
import uuid

import httpx
from fastapi.testclient import TestClient

from snatchguard.central.app import create_app
from snatchguard.central.notify import TelegramNotifier, caption

KEY = {"X-API-Key": "test-key"}


class FakeTelegram:
    def __init__(self, fail_first: int = 0):
        self.calls: list[tuple[str, bytes]] = []
        self.fail_first = fail_first
        self.next_id = 100

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(502, json={"ok": False, "description": "bad gateway"})
        self.calls.append((method, request.read()))
        self.next_id += 1
        return httpx.Response(200, json={"ok": True, "result": {"message_id": self.next_id}})


def _notifier(tmp_path, fake, **kw):
    return TelegramNotifier("TOKEN", ["42"], log_path=tmp_path / "notify.jsonl", retry_wait_s=0.0,
                            client=httpx.Client(transport=httpx.MockTransport(fake)), **kw)


def _event(level="CONFIRMED", **kw):
    return {"event_id": str(uuid.uuid4()), "type": "snatch", "level": level, "camera_name": "Gate",
            "occurred_at": "2026-10-24T10:00:00+00:00", "detected_at": "2026-10-24T10:00:02.5+00:00",
            "received_at": "2026-10-24T10:00:03+00:00", "mode": "on_bike", "score": 0.91, **kw}


def test_caption_says_what_where_and_how_late():
    text = caption(_event(), "http://10.0.0.5:8000")
    assert "CHAIN SNATCHING - Gate" in text and "on bike" in text and "score 0.91" in text
    assert "alert 2.5 s after the contact" in text and "http://10.0.0.5:8000/" in text
    assert "Check the clip" in caption(_event("SUSPECT"))


def test_alert_with_snapshot_then_clip_as_reply(tmp_path):
    fake = FakeTelegram()
    n = _notifier(tmp_path, fake)
    snap, clip = tmp_path / "s.jpg", tmp_path / "clip.mp4"
    snap.write_bytes(b"\xff\xd8jpeg")
    clip.write_bytes(b"mp4data")
    ev = _event()
    n.event(ev, snap)
    n.clip(ev["event_id"], clip)
    n.close()
    assert [m for m, _ in fake.calls] == ["sendPhoto", "sendVideo"]
    assert b"jpeg" in fake.calls[0][1] and b"reply_to_message_id" in fake.calls[1][1] and b"101" in fake.calls[1][1]
    rec = json.loads((tmp_path / "notify.jsonl").read_text())
    assert rec["delivered"] and rec["contact_to_decision_s"] == 2.5 and rec["decision_to_central_s"] == 0.5


def test_retries_after_a_network_error_and_filters_levels(tmp_path):
    fake = FakeTelegram(fail_first=2)
    n = _notifier(tmp_path, fake, levels=("CONFIRMED",))
    n.event(_event("SUSPECT"))
    n.event(_event("CONFIRMED"))
    n.close()
    assert [m for m, _ in fake.calls] == ["sendMessage"]      # SUSPECT not sent; CONFIRMED sent on the 3rd try


def test_upgrade_replies_to_the_earlier_alert(tmp_path):
    fake = FakeTelegram()
    n = _notifier(tmp_path, fake)
    first = _event("SUSPECT")
    n.event(first)
    n.event(_event("CONFIRMED", supersedes=first["event_id"]))
    n.close()
    assert b"reply_to_message_id" not in fake.calls[0][1] and b"101" in fake.calls[1][1]


def test_central_sends_each_new_alert_once(tmp_path):
    fake = FakeTelegram()
    n = _notifier(tmp_path, fake)
    ev = _event()
    with TestClient(create_app(tmp_path, {"test-key"}, notifier=n)) as c:
        files = {"snapshot": ("s.jpg", b"\xff\xd8jpeg", "image/jpeg")}
        for _ in range(2):                                      # the edge retries: same event id
            c.post("/api/v1/events", data={"meta": json.dumps(ev)}, files=files, headers=KEY)
        c.post(f"/api/v1/events/{ev['event_id']}/clip", files={"clip": ("c.mp4", b"mp4", "video/mp4")}, headers=KEY)
    assert [m for m, _ in fake.calls] == ["sendPhoto", "sendVideo"]

"""Central API: auth, idempotent events, clips, heartbeats, reviews, offline watchdog."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from snatchguard.central.app import create_app

KEY = {"X-API-Key": "test-key"}


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path, {"test-key"})) as c:
        yield c


def _event(level="CONFIRMED", **kw):
    return {"event_id": str(uuid.uuid4()), "type": "snatch", "level": level, "node_id": "n1", "camera_id": "c1",
            "camera_name": "Cam 1", "occurred_at": "2026-01-01T00:00:00+00:00", "score": 0.9,
            "confidence": "high", "lighting": "ir", "evidence": [], **kw}


def test_rejects_bad_key(client):
    r = client.post("/api/v1/events", data={"meta": json.dumps(_event())}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_event_is_idempotent_and_listed(client):
    ev = _event()
    files = {"snapshot": ("s.jpg", b"\xff\xd8fakejpeg", "image/jpeg")}
    r1 = client.post("/api/v1/events", data={"meta": json.dumps(ev)}, files=files, headers=KEY)
    r2 = client.post("/api/v1/events", data={"meta": json.dumps(ev)}, files=files, headers=KEY)
    assert r1.json() == {"ok": True, "duplicate": False}
    assert r2.json()["duplicate"] is True
    events = client.get("/api/v1/events").json()
    assert len(events) == 1 and events[0]["level"] == "CONFIRMED" and events[0]["lighting"] == "ir"
    assert client.get(events[0]["snapshot"]).status_code == 200


def test_clip_needs_event_first_then_attaches(client):
    ev = _event()
    clip = {"clip": ("c.webm", b"webmdata" * 100, "video/webm")}
    assert client.post(f"/api/v1/events/{ev['event_id']}/clip", files=clip, headers=KEY).status_code == 404
    client.post("/api/v1/events", data={"meta": json.dumps(ev)}, headers=KEY)
    assert client.post(f"/api/v1/events/{ev['event_id']}/clip", files=clip, headers=KEY).status_code == 200
    assert client.get(f"/api/v1/events/{ev['event_id']}").json()["clip"].endswith("clip.webm")


def test_path_traversal_blocked(client):
    ev = _event(event_id="../../etc")
    r = client.post("/api/v1/events", data={"meta": json.dumps(ev)},
                    files={"snapshot": ("s.jpg", b"x", "image/jpeg")}, headers=KEY)
    assert r.status_code == 400


def test_review_and_stats(client):
    ev = _event()
    client.post("/api/v1/events", data={"meta": json.dumps(ev)}, headers=KEY)
    client.post("/api/v1/events", data={"meta": json.dumps(_event("SUSPECT"))}, headers=KEY)
    r = client.post(f"/api/v1/events/{ev['event_id']}/review", json={"status": "false_alarm", "note": "kids playing"})
    assert r.json()["status"] == "false_alarm"
    s = client.get("/api/v1/stats").json()
    assert s["confirmed"] == 1 and s["suspect"] == 1 and s["reviewed_false"] == 1 and s["open"] == 1


def test_heartbeat_and_thumbnail(client):
    import base64
    hb = {"node_id": "n1", "site": "s", "cameras": [{"id": "c1", "name": "Cam", "status": "online",
                                                    "thumbnail": base64.b64encode(b"jpg").decode()}]}
    assert client.post("/api/v1/heartbeat", json=hb, headers=KEY).status_code == 200
    cams = client.get("/api/v1/cameras").json()["cameras"]
    assert cams[0]["status"] == "online" and "thumbnail" not in cams[0]["payload"]
    assert client.get("/api/v1/cameras/n1/c1/thumbnail.jpg").content == b"jpg"


def test_silent_node_raises_offline_alert(tmp_path, monkeypatch):
    import snatchguard.central.app as appmod
    monkeypatch.setattr(appmod, "OFFLINE_AFTER_S", 0.5)
    with TestClient(create_app(tmp_path, {"test-key"})) as c:
        c.post("/api/v1/heartbeat", json={"node_id": "n9", "cameras": [{"id": "c1", "status": "online"}]}, headers=KEY)
        import time
        deadline = time.time() + 12
        while time.time() < deadline:
            evs = c.get("/api/v1/events").json()
            if any(e["level"] == "OFFLINE" for e in evs):
                break
            time.sleep(0.5)
        assert any(e["level"] == "OFFLINE" and e["node_id"] == "n9" for e in evs)
        assert c.get("/api/v1/cameras").json()["cameras"][0]["status"] == "offline"
        # heartbeat again -> back online event
        c.post("/api/v1/heartbeat", json={"node_id": "n9", "cameras": [{"id": "c1", "status": "online"}]}, headers=KEY)
        assert any(e["level"] == "ONLINE" for e in c.get("/api/v1/events").json())

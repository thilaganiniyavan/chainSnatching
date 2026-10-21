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


def test_dashboard_password(tmp_path):
    import base64 as b64
    with TestClient(create_app(tmp_path, {"test-key"}, dashboard_password="s3cret")) as c:
        assert c.get("/").status_code == 401
        assert c.get("/api/v1/events").status_code == 401
        good = {"Authorization": "Basic " + b64.b64encode(b"operator:s3cret").decode()}
        bad = {"Authorization": "Basic " + b64.b64encode(b"operator:nope").decode()}
        assert c.get("/api/v1/events", headers=good).status_code == 200
        assert c.get("/api/v1/events", headers=bad).status_code == 401
        # edge nodes keep working with their API key
        assert c.post("/api/v1/heartbeat", json={"node_id": "n1", "cameras": []}, headers=KEY).status_code == 200
        assert c.get("/healthz").status_code == 200


def test_new_payload_fields_round_trip(client):
    ev = _event(level="SUSPECT", source="model", model_score=0.93, rule_score=None, mode="unknown",
                supersedes=None)
    client.post("/api/v1/events", data={"meta": json.dumps(ev)}, headers=KEY)
    got = client.get(f"/api/v1/events/{ev['event_id']}").json()
    assert got["payload"]["source"] == "model" and got["payload"]["model_score"] == 0.93
    old = _event()                                       # an edge without the learned scorer still works
    assert client.post("/api/v1/events", data={"meta": json.dumps(old)}, headers=KEY).json()["ok"]


def test_live_risk_endpoint(client):
    assert client.post("/api/v1/live", json={"node_id": "n1", "cameras": [{"id": "c1", "risk": 0.5}]},
                       headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/api/v1/live", json={"cameras": []}, headers=KEY).status_code == 422
    assert client.post("/api/v1/live", json={"node_id": "n1", "cameras": {"id": "c1"}}, headers=KEY).status_code == 422
    r = client.post("/api/v1/live", json={"node_id": "n1", "cameras": [
        {"id": "c1", "risk": 1.7, "learned": "on"}, {"id": "c2", "risk": "high"}, {"id": "../x", "risk": 0.2}]},
        headers=KEY)
    assert r.json() == {"ok": True, "cameras": 2}
    live = {c["camera_id"]: c for c in client.get("/api/v1/live").json()["cameras"]}
    assert live["c1"]["risk"] == 1.0 and live["c1"]["learned"] == "on"
    assert live["c2"]["risk"] is None and "../x" not in live

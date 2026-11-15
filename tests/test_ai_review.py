"""AI review of snatch alerts: held until reviewed, rejected ones hidden, confirmed ones badged, failures shown."""
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from snatchguard.central import review
from snatchguard.central.app import create_app

KEY = {"x-api-key": "k"}


class FakeReviewer:
    def __init__(self, states):
        self.states, self.calls = states, []

    def review(self, path, at=None):
        self.calls.append((path.name, at))
        return {"state": self.states[path.parent.name]}


def post_event(client, eid, **extra):
    meta = {"event_id": eid, "type": "snatch", "level": "CONFIRMED", "node_id": "footage-x", "camera_id": "c",
            "clip_event_at_s": 6, **extra}
    r = client.post("/api/v1/events", data={"meta": json.dumps(meta)},
                    files={"snapshot": ("s.jpg", cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes())},
                    headers=KEY)
    assert r.status_code == 200


def wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def visible_ids(client):
    return {e["id"] for e in client.get("/api/v1/events").json()}


def test_alerts_are_held_then_shown_or_hidden(tmp_path):
    fake = FakeReviewer({"a": "confirmed", "b": "rejected", "c": "unavailable"})
    app = create_app(tmp_path, {"k"}, dashboard_password="", notifier=None, reviewer=fake)
    with TestClient(app) as client:
        for eid in "abc":
            post_event(client, eid)
        assert visible_ids(client) == set()                      # held while pending
        assert client.get("/api/v1/events/a").status_code == 404
        for eid in "abc":
            client.post(f"/api/v1/events/{eid}/clip", files={"clip": ("clip.webm", b"x")}, headers=KEY)
        assert wait(lambda: len(fake.calls) == 3)
        assert wait(lambda: visible_ids(client) == {"a", "c"})   # rejected stays hidden; failure is shown
        assert fake.calls[0] == ("clip.webm", 6)
        a = client.get("/api/v1/events/a").json()
        assert a["payload"]["ai_review"]["state"] == "confirmed"
        assert client.get("/api/v1/events/b").status_code == 404
        assert client.get("/api/v1/stats").json()["confirmed"] == 2


def test_snapshot_is_reviewed_when_no_clip_arrives(tmp_path, monkeypatch):
    monkeypatch.setenv("SNATCHGUARD_REVIEW_WAIT_S", "0.2")
    fake = FakeReviewer({"d": "confirmed"})
    app = create_app(tmp_path, {"k"}, dashboard_password="", notifier=None, reviewer=fake)
    with TestClient(app) as client:
        post_event(client, "d")
        assert wait(lambda: visible_ids(client) == {"d"})
        assert fake.calls == [("snapshot.jpg", 6)]


def test_without_reviewer_alerts_publish_immediately(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("SNATCHGUARD_AI_REVIEW", raising=False)
    app = create_app(tmp_path, {"k"}, dashboard_password="", notifier=None)
    with TestClient(app) as client:
        post_event(client, "e")
        assert visible_ids(client) == {"e"}
        assert "ai_review" not in client.get("/api/v1/events/e").json()["payload"]


def _clip(path, n=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), (i * 8) % 256, np.uint8))
    w.release()
    return path


class FakeClient:
    def __init__(self, reply=None, error=None, text=None):
        self.reply, self.error, self.text, self.kwargs = reply, error, text, None
        self.models = SimpleNamespace(generate_content=self.create)

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text if self.text is not None else json.dumps(self.reply))


def test_reviewer_sends_the_video_and_applies_confidence(tmp_path):
    clip = _clip(tmp_path / "clip.mp4")
    fc = FakeClient({"snatch": True, "confidence": 0.9, "reason": "grab"})
    out = review.AIReviewer(fc).review(clip, 6)
    media = fc.kwargs["contents"][0]
    assert media.inline_data.mime_type == "video/mp4" and media.inline_data.data == clip.read_bytes()
    assert media.video_metadata.fps == review.VIDEO_FPS and out["state"] == "confirmed"
    assert fc.kwargs["config"].response_mime_type == "application/json"
    assert review.AIReviewer(FakeClient({"snatch": True, "confidence": 0.3, "reason": "?"})).review(clip)["state"] == "rejected"
    assert review.AIReviewer(FakeClient({"snatch": False, "confidence": 0.9, "reason": "walk"})).review(clip)["state"] == "rejected"
    assert review.AIReviewer(FakeClient(error=RuntimeError("down"))).review(clip)["state"] == "unavailable"
    assert review.AIReviewer(FakeClient(text="")).review(clip)["state"] == "unavailable"
    assert review.AIReviewer(FakeClient()).review(tmp_path / "missing.mp4")["state"] == "unavailable"


def test_camera_token_works_without_dashboard_password(tmp_path):
    app = create_app(tmp_path, {"k"}, dashboard_password="", notifier=None)
    with TestClient(app) as client:
        assert client.get("/api/v1/browsercam/token").json()["token"]


def test_confirmed_alert_is_upgraded_and_one_incident_shown_once(tmp_path):
    fake = FakeReviewer({"f": "confirmed", "g": "confirmed", "h": "confirmed"})
    app = create_app(tmp_path, {"k"}, dashboard_password="", notifier=None, reviewer=fake)
    with TestClient(app) as client:
        post_event(client, "f", level="SUSPECT", video_time_s=10.0)
        post_event(client, "g", level="SUSPECT", video_time_s=14.0)    # same incident
        post_event(client, "h", level="SUSPECT", video_time_s=60.0)    # a separate one
        for eid in "fgh":
            client.post(f"/api/v1/events/{eid}/clip", files={"clip": ("clip.webm", b"x")}, headers=KEY)
            assert wait(lambda: len(fake.calls) == "fgh".index(eid) + 1)
        assert wait(lambda: visible_ids(client) == {"f", "h"})
        assert client.get("/api/v1/events/f").json()["level"] == "CONFIRMED"


def test_ai_review_runs_the_pipeline_at_its_most_sensitive(tmp_path, monkeypatch):
    import yaml
    from snatchguard.central.footage import FootageRunner, SENSITIVITY

    class Process:
        def __init__(self, *a, **k): self.code = None
        def poll(self): return self.code
    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen", Process)
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["m.pt"])
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    job = FootageRunner(tmp_path, "k").start("v.mp4", "v", 8000, model="m.pt")
    v = yaml.safe_load((tmp_path / job["id"] / "config.yaml").read_text())["verifier"]
    assert v["threshold"] == SENSITIVITY["maximum"] and v["merge_s"] == 3


def test_trimmed_video_is_cut_from_the_upload_boxed_and_reviewed(tmp_path, monkeypatch):
    from snatchguard.central import trim
    monkeypatch.setenv("SNATCHGUARD_BOX_WAIT_S", "20")
    src = _clip(tmp_path / "upload.mp4", n=300)                         # 30 s at 10 fps, 64x48
    fake = FakeReviewer({"u": "confirmed"})
    app = create_app(tmp_path / "c", {"k"}, dashboard_password="", notifier=None, reviewer=fake)
    boxes = {"roles": {"1": "actor"}, "frames": [
        {"t": round(t * 0.2, 2), "wall": None, "boxes": [[1, "person", 0.25, 0.25, 0.75, 0.75], [2, "person", 0, 0, 0.1, 0.1]]}
        for t in range(150)]}
    with TestClient(app) as client:
        app.state.runner.sources["footage-x"] = {"kind": "file", "path": str(src)}
        post_event(client, "u", video_time_s=15.0)
        time.sleep(0.5)
        client.post("/api/v1/events/u/clip", files={"clip": ("clip.webm", b"x"),
                                                    "boxes": ("b.json", json.dumps(boxes).encode())}, headers=KEY)
        assert wait(lambda: visible_ids(client) == {"u"}, timeout=30)
        ev = client.get("/api/v1/events/u").json()
        assert ev["clip"] == "/media/u/trimmed_boxes.mp4"
        assert fake.calls == [("trimmed.mp4", 10.0)]                    # the AI sees the clean video
    cap = cv2.VideoCapture(str(tmp_path / "c/media/u/trimmed_boxes.mp4"))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    assert abs(n - (trim.PRE_S + trim.POST_S) * 10) <= 3
    b, g, r = (int(v) for v in frames[50][12, 32])                     # 5 s in: amber DETECTED box, top edge
    assert r > 150 and g > 100 and b < 120, (b, g, r)
    b, g, r = (int(v) for v in frames[100][12, 32])                    # the contact: red SNATCH box over it
    assert r > 150 and g < 100 and b < 100, (b, g, r)


def test_boxes_are_interpolated_between_analysed_frames():
    from snatchguard.central.trim import _boxes_at
    frames = [{"t": 0.0, "boxes": [[1, "person", 0.0, 0.0, 0.2, 0.2]]}, {"t": 1.0, "boxes": [[1, "person", 0.2, 0.2, 0.4, 0.4]]}]
    (_cls, box), = _boxes_at(frames, "t", 0.5).values()
    assert [round(v, 3) for v in box] == [0.1, 0.1, 0.3, 0.3]
    assert _boxes_at(frames, "t", 5.0) == {}


def test_frames_to_mp4_keeps_capture_rate(tmp_path):
    from snatchguard.central import trim
    jpg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()
    out = trim.frames_to_mp4([(i / 10, jpg) for i in range(50)], tmp_path / "t.mp4")
    cap = cv2.VideoCapture(str(out))
    assert round(cap.get(cv2.CAP_PROP_FPS)) == 10 and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 50

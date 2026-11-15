"""Live-camera configuration and dashboard API integration."""
import json

import yaml
from fastapi.testclient import TestClient

from snatchguard.central.app import create_app
from snatchguard.central.footage import FootageRunner


def test_live_runner_configures_camera_and_verifier(tmp_path, monkeypatch):
    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
            self.code = None
        def poll(self):
            return self.code
        def terminate(self):
            self.code = -15

    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen", Process)
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["trusted.pt"])
    runner = FootageRunner(tmp_path, "test-key")
    job = runner.start("rtsp://camera/stream", "Entrance", 8080, live=True, model="trusted.pt", site="Campus")
    cfg = yaml.safe_load((tmp_path / job["id"] / "config.yaml").read_text())
    assert cfg["site"] == "Campus"
    assert cfg["cameras"][0]["source"] == "rtsp://camera/stream"
    assert cfg["verifier"]["enabled"] and cfg["verifier"]["model"].endswith("/models/verifier/trusted.pt")
    assert cfg["central"]["url"] == "http://127.0.0.1:8080"
    assert "--until-done" not in runner.process.args
    assert runner.stop()["state"] == "stopped"


def test_live_api_validates_source_model_and_busy_state(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["trusted.pt"])
    def start(self, source, name, port, **kwargs):
        if kwargs["model"] != "trusted.pt":
            raise ValueError("Select an available verifier model.")
        if calls:
            raise RuntimeError("A video is already being processed.")
        calls.append((source, name, port, kwargs))
        return {"state": "running"}
    monkeypatch.setattr(FootageRunner, "start", start)
    with TestClient(create_app(tmp_path, {"test-key"})) as client:
        assert client.get("/footage").status_code == 200
        assert client.get("/api/v1/footage/models").json() == {"models": ["trusted.pt"]}
        url = "/api/v1/footage/live"
        assert client.post(url, json={"source": "../../video", "model": "trusted.pt"}).status_code == 422
        assert client.post(url, json={"source": "0", "model": "../bad.pt"}).status_code == 422
        assert client.post(url, json={"source": "0", "name": "Front", "model": "trusted.pt"}).status_code == 200
        assert calls[0][0:2] == ("0", "Front") and calls[0][3]["live"] is True
        assert client.post(url, json={"source": "0", "model": "trusted.pt"}).status_code == 409


def test_upload_runner_uses_selected_model_and_rejects_unknown(tmp_path, monkeypatch):
    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
        def poll(self):
            return None

    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen", Process)
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["trusted.pt"])
    runner = FootageRunner(tmp_path, "test-key")
    try:
        runner.start(str(tmp_path / "v.mp4"), "Upload", 8080, model="../bad.pt")
        raise AssertionError("unknown model accepted")
    except ValueError:
        pass
    job = runner.start(str(tmp_path / "v.mp4"), "Upload", 8080, model="trusted.pt")
    cfg = yaml.safe_load((tmp_path / job["id"] / "config.yaml").read_text())
    assert cfg["verifier"]["enabled"] and cfg["verifier"]["model"].endswith("/models/verifier/trusted.pt")
    assert "--until-done" in runner.process.args


def test_sensitivity_sets_verifier_threshold(tmp_path, monkeypatch):
    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
        def poll(self):
            return None

    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen", Process)
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["trusted.pt"])
    from snatchguard.central.footage import SENSITIVITY
    for level, thr in SENSITIVITY.items():
        runner = FootageRunner(tmp_path / level, "k")
        job = runner.start(str(tmp_path / "v.mp4"), "Upload", 8080, model="trusted.pt", sensitivity=level)
        cfg = yaml.safe_load((tmp_path / level / job["id"] / "config.yaml").read_text())
        assert cfg["verifier"].get("threshold") == thr and job["sensitivity"] == level
    try:
        FootageRunner(tmp_path / "bad", "k").start(str(tmp_path / "v.mp4"), "U", 8080, model="trusted.pt", sensitivity="ultra")
        raise AssertionError("unknown sensitivity accepted")
    except ValueError:
        pass


def test_output_lists_one_clip_per_incident(tmp_path, monkeypatch):
    monkeypatch.setattr(FootageRunner, "status", lambda self: {"id": "job1", "state": "completed", "name": "v.mp4"})
    with TestClient(create_app(tmp_path, {"k"})) as client:
        def post(eid, t, level, sup=None):
            ev = {"event_id": eid, "type": "snatch", "node_id": "footage-job1", "camera_id": "camera-01",
                  "occurred_at": "2026-11-15T00:00:00+00:00", "detected_at": "2026-11-15T00:00:01+00:00",
                  "level": level, "score": 0.9, "video_time_s": t}
            if sup:
                ev["supersedes"] = sup
            assert client.post("/api/v1/events", data={"meta": json.dumps(ev)}, headers={"x-api-key": "k"}).status_code == 200
        post("a", 12.0, "SUSPECT"); post("b", 13.0, "CONFIRMED", sup="a"); post("c", 40.0, "SUSPECT")
        other = {"event_id": "z", "type": "snatch", "node_id": "footage-other", "camera_id": "c", "level": "SUSPECT",
                 "occurred_at": "2026-11-15T00:00:00+00:00", "detected_at": "2026-11-15T00:00:00+00:00"}
        client.post("/api/v1/events", data={"meta": json.dumps(other)}, headers={"x-api-key": "k"})
        out = client.get("/api/v1/footage/output").json()
        assert [i["id"] for i in out["incidents"]] == ["b", "c"]


def test_default_model_used_when_none_given(tmp_path, monkeypatch):
    class Process:
        def __init__(self, args, **kwargs):
            self.args = args
        def poll(self):
            return None

    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen", Process)
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["only.pt"])
    runner = FootageRunner(tmp_path, "k")
    job = runner.start("0", "Live CCTV", 8080, live=True)
    cfg = yaml.safe_load((tmp_path / job["id"] / "config.yaml").read_text())
    assert job["model"] == "only.pt" and cfg["verifier"]["model"].endswith("/models/verifier/only.pt")
    assert cfg["verifier"]["threshold"] == 0.3 and job["node_id"] == cfg["node_id"]
    monkeypatch.setattr(FootageRunner, "models", lambda self: [])
    try:
        FootageRunner(tmp_path / "none", "k").start("0", "Live CCTV", 8080, live=True)
        raise AssertionError("started without a model")
    except ValueError:
        pass


def test_browser_camera_login_and_invalid_tokens(tmp_path):
    import base64
    from starlette.websockets import WebSocketDisconnect
    import pytest
    app = create_app(tmp_path, {"edge-key"}, dashboard_password="pw")
    login = {"authorization": "Basic " + base64.b64encode(b"operator:pw").decode()}
    with TestClient(app) as client:
        assert client.get("/api/v1/browsercam/token").status_code == 401
        assert client.get("/api/v1/browsercam/stream.mjpg?key=wrong").status_code == 401
        session = client.get("/api/v1/browsercam/token", headers=login).json()
        assert session["id"] and session["token"]
        assert client.get("/api/v1/browsercam/stream.mjpg?key=edge-key&id=missing").status_code == 404
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/v1/browsercam/ws?id=missing&token=bad"):
                pass


def test_stopped_job_that_hangs_is_force_closed_and_never_blocks_a_new_one(tmp_path, monkeypatch):
    import subprocess, sys, time
    monkeypatch.setattr(FootageRunner, "models", lambda self: ["m.pt"])
    real = subprocess.Popen
    hang = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    monkeypatch.setattr("snatchguard.central.footage.subprocess.Popen",
                        lambda args, **kw: real([sys.executable, "-c", hang], start_new_session=True))
    runner = FootageRunner(tmp_path, "k")
    monkeypatch.setattr(FootageRunner, "STOP_GRACE_S", 0.5)
    first = runner.start("v.mp4", "v", 8000, model="m.pt")
    time.sleep(0.5)
    runner.stop()
    proc = runner.process
    deadline = time.time() + 8
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)
    assert proc.poll() is not None                       # force-closed after the grace period
    second = runner.start("v.mp4", "v", 8000, model="m.pt")
    runner.stop()                                        # stop requested + still alive: a new start is not blocked
    third = runner.start("v.mp4", "v", 8000, model="m.pt")
    assert len({first["id"], second["id"], third["id"]}) == 3
    runner._kill(runner.process)

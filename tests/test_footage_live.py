"""Live-camera configuration and dashboard API integration."""
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
    job = runner.start("rtsp://camera/stream", "Entrance", 8080, live=True, model="trusted.pt")
    cfg = yaml.safe_load((tmp_path / job["id"] / "config.yaml").read_text())
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

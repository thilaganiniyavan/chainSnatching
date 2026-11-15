"""Local, single-job footage runner for the operator dashboard."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import yaml

from snatchguard.core.config import REPO_ROOT, load_yaml


class FootageRunner:
    def __init__(self, directory: Path, key: str):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.process = None
        self.job = None
        self.source = None
        self.stop_requested = False
        self.lock = threading.Lock()

    def status(self):
        if not self.job:
            return {"state": "idle"}
        result = dict(self.job)
        code = self.process.poll()
        result["state"] = ("running" if code is None else
                           ("stopped" if self.stop_requested else ("completed" if code == 0 else "failed")))
        result["exit_code"] = code
        if code not in (None, 0) and not self.stop_requested:
            try:
                message = (self.directory / result["id"] / "run.log").read_text(errors="replace")[-2000:]
                result["message"] = message.replace(self.source, "[camera source]") if self.source else message
            except OSError:
                pass
        return result

    def models(self):
        directory = REPO_ROOT / "models/verifier"
        return sorted(p.name for p in directory.glob("*.pt") if p.is_file() and not p.is_symlink())

    def start(self, source: str, name: str, port: int, demo: bool = False, real_demo: bool = False,
              live: bool = False, model: str | None = None):
        with self.lock:
            return self._start(source, name, port, demo, real_demo, live, model)

    def _start(self, source: str, name: str, port: int, demo: bool, real_demo: bool,
               live: bool, model: str | None):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("A video is already being processed. Stop it before starting another.")
        if live and (not model or model not in self.models()):
            raise ValueError("Select an available verifier model.")
        job_id = uuid.uuid4().hex[:12]
        folder = self.directory / job_id
        folder.mkdir()
        cfg = load_yaml(REPO_ROOT / "configs/edge.yaml")
        cfg["node_id"] = "footage-" + job_id
        cfg["site"] = "Live camera" if live else ("Scripted demonstration" if demo else ("Real CCTV demonstration · training clip" if real_demo else "Uploaded footage"))
        if not demo:
            cfg["verifier"]["enabled"] = True
        if model:
            cfg["verifier"]["model"] = str(REPO_ROOT / "models/verifier" / model)
        cfg["central"] = {"url": f"http://127.0.0.1:{port}", "api_key": self.key, "heartbeat_s": 2}
        cfg["outbox"] = str(folder / "outbox.db")
        cfg["evidence"]["dir"] = str(folder / "evidence")
        cfg["cameras"] = [{"id": "camera-01", "name": name, "source": source, "loop": demo,
                           "realtime": True, "lighting": "auto", "simulate_night": "none"}]
        config_path = folder / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        os.chmod(config_path, 0o600)
        log = (folder / "run.log").open("w")
        os.chmod(folder / "run.log", 0o600)
        try:
            self.process = subprocess.Popen([sys.executable, "-m", "snatchguard.edge", "--config", str(config_path),
                                             *([] if demo or live else ["--until-done"])], cwd=REPO_ROOT,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        self.source = source if live else None
        self.stop_requested = False
        self.job = {"id": job_id, "name": name, "kind": "live camera" if live else ("scripted demo" if demo else ("real CCTV demo · training clip" if real_demo else "uploaded video")), "model": model}
        return self.status()

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.stop_requested = True
                self.process.terminate()
            return self.status()

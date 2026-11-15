"""Local, single-job footage runner for the operator dashboard."""
from __future__ import annotations

import os
import subprocess
import sys
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

    def status(self):
        if not self.job:
            return {"state": "idle"}
        result = dict(self.job)
        code = self.process.poll()
        result["state"] = "running" if code is None else ("completed" if code == 0 else "failed")
        result["exit_code"] = code
        if code not in (None, 0):
            try:
                result["message"] = (self.directory / result["id"] / "run.log").read_text(errors="replace")[-2000:]
            except OSError:
                pass
        return result

    def start(self, source: str, name: str, port: int, demo: bool = False, real_demo: bool = False):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("A video is already being processed. Stop it before starting another.")
        job_id = uuid.uuid4().hex[:12]
        folder = self.directory / job_id
        folder.mkdir()
        cfg = load_yaml(REPO_ROOT / "configs/edge.yaml")
        cfg["node_id"] = "footage-" + job_id
        cfg["site"] = "Scripted demonstration" if demo else ("Real CCTV demonstration · training clip" if real_demo else "Uploaded footage")
        if not demo:
            cfg["verifier"]["enabled"] = True
        cfg["central"] = {"url": f"http://127.0.0.1:{port}", "api_key": self.key, "heartbeat_s": 2}
        cfg["outbox"] = str(folder / "outbox.db")
        cfg["evidence"]["dir"] = str(folder / "evidence")
        cfg["cameras"] = [{"id": "camera-01", "name": name, "source": source, "loop": demo,
                           "realtime": True, "lighting": "auto", "simulate_night": "none"}]
        config_path = folder / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))
        os.chmod(config_path, 0o600)
        log = (folder / "run.log").open("w")
        try:
            self.process = subprocess.Popen([sys.executable, "-m", "snatchguard.edge", "--config", str(config_path),
                                             *([] if demo else ["--until-done"])], cwd=REPO_ROOT,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        self.job = {"id": job_id, "name": name, "kind": "scripted demo" if demo else ("real CCTV demo · training clip" if real_demo else "uploaded video")}
        return self.status()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        return self.status()

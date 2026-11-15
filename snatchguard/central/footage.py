"""Local, single-job footage runner for the operator dashboard."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import yaml

from snatchguard.central import review as ai_review
from snatchguard.core.config import REPO_ROOT, load_yaml

# Verifier alert threshold per sensitivity (6-view scores; chosen on the Stage 2 validation split, not the test set;
# cleaned/stage6_dashboard/curve_val.json). "standard" keeps the threshold stored in the model. Alerts below the
# model's CONFIRMED threshold arrive as SUSPECT. Higher sensitivity catches more incidents and raises more false alarms.
SENSITIVITY = {"standard": None, "high": 0.3, "maximum": 0.05}


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
        self.sources: dict[str, dict] = {}     # node id -> where its video comes from (for trimmed detected videos)

    def source_for(self, node_id: str | None) -> dict | None:
        return self.sources.get(node_id or "")

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

    def default_model(self) -> str | None:
        """The verifier the dashboard uses without asking: the one named in configs/edge.yaml if installed, else the
        newest checkpoint in models/verifier."""
        names = self.models()
        configured = Path(str(load_yaml(REPO_ROOT / "configs/edge.yaml").get("verifier", {}).get("model", ""))).name
        if configured in names:
            return configured
        directory = REPO_ROOT / "models/verifier"
        return max(names, key=lambda n: (directory / n).stat().st_mtime if (directory / n).exists() else 0.0) if names else None

    def start(self, source: str, name: str, port: int, demo: bool = False, real_demo: bool = False,
              live: bool = False, model: str | None = None, sensitivity: str = "high", site: str | None = None):
        with self.lock:
            return self._start(source, name, port, demo, real_demo, live, model, sensitivity, site)

    def _start(self, source: str, name: str, port: int, demo: bool, real_demo: bool,
               live: bool, model: str | None, sensitivity: str = "high", site: str | None = None):
        if self.process is not None and self.process.poll() is None:
            if not self.stop_requested:
                raise RuntimeError("A video is already being processed. Stop it before starting another.")
            self._kill(self.process)             # a stopped job still closing down must not block a new one
        if model is None and not demo:
            model = self.default_model()
            if model is None:
                raise ValueError("No detection model is installed (models/verifier is empty).")
        if model and model not in self.models():
            raise ValueError("Select an available verifier model.")
        if sensitivity not in SENSITIVITY:
            raise ValueError("Select a sensitivity: standard, high or maximum.")
        job_id = uuid.uuid4().hex[:12]
        folder = self.directory / job_id
        folder.mkdir()
        cfg = load_yaml(REPO_ROOT / "configs/edge.yaml")
        cfg["node_id"] = "footage-" + job_id
        cfg["site"] = (site or "Live camera") if live else ("Scripted demonstration" if demo else ("Real CCTV demonstration · training clip" if real_demo else "Uploaded footage"))
        if not demo:
            cfg["verifier"]["enabled"] = True
        if model:
            cfg["verifier"]["model"] = str(REPO_ROOT / "models/verifier" / model)
        if SENSITIVITY[sensitivity] is not None:
            cfg["verifier"]["threshold"] = SENSITIVITY[sensitivity]
        if ai_review.enabled() and not demo:
            # The AI review removes false alarms, so the pipeline can run at its most sensitive setting (fewer missed
            # snatches) and send every candidate: incidents are merged after the review, not before (a rejected first
            # candidate must not swallow a real grab seconds later).
            cfg["verifier"]["threshold"] = min(cfg["verifier"].get("threshold") or 1.0, SENSITIVITY["maximum"])
            cfg["verifier"]["merge_s"] = 3
        cfg["central"] = {"url": os.environ.get("SNATCHGUARD_INTERNAL_URL", f"http://127.0.0.1:{port}").rstrip("/"), "api_key": self.key, "heartbeat_s": 1}
        cfg["thumbnail_width"] = 640                          # the live view on the footage page
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
        if "/api/v1/browsercam/stream.mjpg" in source:
            self.sources[cfg["node_id"]] = {"kind": "browsercam"}
        elif not live and Path(source).is_file():
            self.sources[cfg["node_id"]] = {"kind": "file", "path": source}
        self.stop_requested = False
        self.job = {"id": job_id, "name": name, "kind": "live camera" if live else ("scripted demo" if demo else ("real CCTV demo · training clip" if real_demo else "uploaded video")), "model": model,
                    "sensitivity": sensitivity, "node_id": cfg["node_id"], "camera_id": "camera-01"}
        return self.status()

    STOP_GRACE_S = 20.0

    @staticmethod
    def _kill(process) -> None:
        """Force-close a job and everything it started (it runs in its own process group)."""
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.stop_requested = True
                self.process.terminate()
                process = self.process

                def force():                     # a job that hangs while shutting down is closed after a grace period
                    try:
                        process.wait(timeout=self.STOP_GRACE_S)
                    except subprocess.TimeoutExpired:
                        self._kill(process)
                    except Exception:  # noqa: BLE001
                        pass
                threading.Thread(target=force, daemon=True).start()
            return self.status()

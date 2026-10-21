"""Per-camera processing loop.

    frame -> lighting mode -> enhancement -> detect + track -> scene/pairs
          -> pose on the closest pairs -> rule engine -> alert / clip / tamper
    frame -> learned snatch scorer (own thread, newest frame only) -/
             rule decisions + window scores -> fusion (evaluation/fusion.py, same code as offline)

Runs in its own process (see node.py), writes alerts to the shared outbox and
reports health to the supervisor through a queue.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from snatchguard.core.config import REPO_ROOT, RuleConfig
from snatchguard.core.types import Detection, FrameObservation, Level, Lighting
from snatchguard.edge.evidence import EvidenceRecorder
from snatchguard.edge.outbox import Outbox
from snatchguard.edge.source import Frame, LiveSource
from snatchguard.edge.tamper import TamperDetector
from snatchguard.rules.engine import Decision, RuleEngine
from snatchguard.sim.night import make_transform

log = logging.getLogger(__name__)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _path(p: str) -> str:
    return str(p if Path(p).is_absolute() else REPO_ROOT / p)


class LearnedRunner(threading.Thread):
    """Runs the learned scorer beside the rule path. It always takes the newest offered frame and drops
    older ones when it falls behind, so a slow model never delays the rules."""

    def __init__(self, scorer, cam_id: str) -> None:
        super().__init__(name=f"learned-{cam_id}", daemon=True)
        self.scorer = scorer
        self.cam_id = cam_id
        self._cv = threading.Condition()
        self._latest: tuple[np.ndarray, float] | None = None
        self._out: list = []
        self._reset = False
        self._halt = False
        self.risk: float | None = None
        self.error: str | None = None
        self.scored = 0
        self.dropped = 0

    def offer(self, img: np.ndarray, t: float) -> None:
        with self._cv:
            if self._latest is not None:
                self.dropped += 1
            self._latest = (img, t)
            self._cv.notify()

    def drain(self) -> list:
        with self._cv:
            out, self._out = self._out, []
        return out

    def reset(self) -> None:
        with self._cv:
            self._reset, self._latest, self._out, self.risk = True, None, [], None

    def stop(self) -> None:
        with self._cv:
            self._halt = True
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while self._latest is None and not self._halt:
                    self._cv.wait(timeout=1.0)
                if self._halt:
                    return
                (img, t), self._latest = self._latest, None
                do_reset, self._reset = self._reset, False
            try:
                if do_reset:
                    self.scorer.reset()
                wins = self.scorer.push(img, t)
            except Exception as exc:              # the rules keep running; fusion falls back to them
                log.exception("[%s] learned scorer failed, continuing with rules only", self.cam_id)
                self.error, self.risk = str(exc)[:200], None
                return
            self.scored += 1
            if wins:
                with self._cv:
                    if not self._reset:
                        self._out.extend(wins)
                        self.risk = float(wins[-1].score)


class SyntheticSource:
    """Plays a scripted scenario as if it were a camera (rendered frames + ground-truth tracks)."""

    synthetic = True

    def __init__(self, scenario_name: str, realtime: bool = True, loop: bool = True, gap_s: float = 20.0) -> None:
        from snatchguard.sim.synthetic import SCENARIOS
        self.factory = SCENARIOS[scenario_name]
        self.realtime, self.loop, self.gap_s = realtime, loop, gap_s
        self.status, self.fps_in, self.src_fps = "online", 8.0, 8.0
        self._gen = self._frames()
        self.scenario = None

    def _frames(self):
        from snatchguard.sim.synthetic import render
        offset, seq, seed = 0.0, 0, 0
        while True:
            self.scenario = sc = self.factory()
            start = time.monotonic()
            last_t = 0.0
            for obs in sc.frames(seed=seed):
                if self.realtime:
                    delay = start + obs.t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                obs.t += offset
                img = render(sc, obs)
                yield Frame(img, obs.t, time.time(), seq), obs
                seq += 1
                last_t = obs.t - offset
            # an empty street between repetitions, so each run is a fresh encounter
            step = 1.0 / sc.fps
            for i in range(int(self.gap_s / step)):
                if self.realtime:
                    time.sleep(step)
                t = offset + last_t + step * (i + 1)
                blank = FrameObservation(t, obs.frame_size, [], {}, sc.lighting)
                yield Frame(render(sc, blank), t, time.time(), seq), blank
                seq += 1
            offset += last_t + self.gap_s + 1.0
            seed += 1
            if not self.loop:
                self.status = "ended"
                return

    def start(self):
        return self

    def stop(self) -> None:
        pass

    def read(self, timeout: float = 1.0):
        try:
            return next(self._gen)
        except StopIteration:
            return None

    @property
    def ended(self) -> bool:
        return self.status == "ended"


class CameraWorker:
    def __init__(self, cam: dict, node: dict, status_queue=None) -> None:
        self.cam = cam
        self.node = node
        self.status_queue = status_queue
        self.id = cam["id"]
        self.name = cam.get("name", self.id)
        proc = node.get("processing", {})
        self.target_fps = float(proc.get("target_fps", 8))
        self.min_fps = float(proc.get("min_fps_warning", 5))
        self.models = node["models"]

        self.tamper = TamperDetector()
        rules = RuleConfig.load(_path(node.get("rules", "configs/rules.yaml")))
        ev = node.get("evidence", {})
        self.recorder = EvidenceRecorder(Path(_path(ev.get("dir", "runtime/edge/evidence"))) / self.id,
                                         ev.get("pre_s", 8), ev.get("post_s", 4), ev.get("clip_width", 960))
        self.outbox = Outbox(_path(node.get("outbox", "runtime/edge/outbox.db")))

        src = str(cam["source"])
        if src.startswith("synthetic:"):
            self.source = SyntheticSource(src.split(":", 1)[1], cam.get("realtime", True), cam.get("loop", True))
            self.pipeline = None
            self.engine = RuleEngine(rules)
        else:
            if src.startswith("playlist:"):
                from snatchguard.edge.source import PlaylistSource, resolve_playlist
                items = resolve_playlist(src, [REPO_ROOT, REPO_ROOT / "data" / "samples" / "snatch1.0" /
                                               "Chain Snatching Videos"])
                self.source = PlaylistSource(items, realtime=cam.get("realtime", True), loop=cam.get("loop", True),
                                             transform=make_transform(cam.get("simulate_night")))
            else:
                if not src.isdigit() and "://" not in src:
                    src = _path(src)
                self.source = LiveSource(src, realtime=cam.get("realtime", True), loop=cam.get("loop", False),
                                         transform=make_transform(cam.get("simulate_night")))
            from snatchguard.edge.pipeline import FramePipeline
            models = {**self.models, "detector": _path(self.models["detector"]),
                      "pose": _path(self.models["pose"]) if self.models.get("pose") else None}
            self.pipeline = FramePipeline(models, _path(node.get("tracker", "configs/tracker.yaml")), rules,
                                          cam.get("lighting", "auto"), cam.get("detect_zone"))
            self.engine = self.pipeline.engine

        self.stats = {"frames": 0, "alerts": 0, "suspects": 0, "dismissed": 0, "fps_proc": 0.0,
                      "det_ms": 0.0, "pose_ms": 0.0, "loop_ms": 0.0}
        self._fps_window: list[float] = []
        self._last_status = 0.0
        self._last_risk_push = 0.0
        self._last_img: np.ndarray | None = None
        self._mode = Lighting.DAY

        self.learned: LearnedRunner | None = None
        self.fuser = None
        self._fusion_params = None
        self._fusion_events: dict[int, str] = {}
        self._learned_dt = 0.125
        self._last_offer = -1e9
        self.learned_state = "off"
        lcfg = node.get("learned") or {}
        if lcfg.get("enabled"):
            self._init_learned(lcfg)

    def _init_learned(self, lcfg: dict) -> None:
        """Learned scorer + fusion, or a single warning and the rules alone (exactly as without it)."""
        model = Path(_path(lcfg.get("model", "models/learned/snatch_scorer.pt")))
        if not model.exists():
            log.warning("[%s] learned scorer enabled but %s does not exist: rules only", self.id, model)
            self.learned_state = "missing"
            return
        try:
            from snatchguard.evaluation.fusion import Fuser, FusionParams
            from snatchguard.learned.scorer import SnatchScorer
            scorer = SnatchScorer(model, device=str(lcfg.get("device", "cpu")), threads=int(lcfg.get("threads", 2)))
            self._fusion_params = FusionParams.load(_path(lcfg["fusion"]) if lcfg.get("fusion") else None)
        except Exception as exc:
            log.warning("[%s] learned scorer unavailable (%s): rules only", self.id, exc)
            self.learned_state = "error"
            return
        self._learned_dt = 1.0 / float(lcfg.get("fps", 8))
        self.fuser = Fuser(self._fusion_params)
        self.learned = LearnedRunner(scorer, self.id)
        self.learned_state = "on"
        log.info("[%s] learned scorer %s on %s, fusion %s", self.id, model.name, lcfg.get("device", "cpu"),
                 self._fusion_params.to_dict())

    def _new_fuser(self) -> None:
        from snatchguard.evaluation.fusion import Fuser
        self.fuser = Fuser(self._fusion_params)
        self._fusion_events = {}

    # ------------------------------------------------------------------
    def run(self, stop_event=None) -> None:
        self.source.start()
        if self.learned is not None:
            self.learned.start()
        log.info("[%s] started (%s)", self.id, self.cam["source"])
        min_dt = 1.0 / self.target_fps
        last_proc = -1e9
        segment = 0
        try:
            while stop_event is None or not stop_event.is_set():
                got = self.source.read(timeout=1.0)
                if got is None:
                    if self.source.ended:
                        break
                    self._report_status(force=False)
                    continue
                frame, truth = got if isinstance(got, tuple) else (got, None)
                if frame.segment != segment and self.pipeline is not None:
                    segment = frame.segment          # next playlist item: a different scene
                    if self.fuser is not None and self._last_img is not None:
                        for a in self.fuser.flush():
                            self._publish_fused(a, Frame(self._last_img, self.engine.scene.t, time.time(), -1),
                                                self._last_img)
                        self.learned.reset()
                        self._new_fuser()
                    self.pipeline.reset()
                    self.engine = self.pipeline.engine
                    self.tamper = TamperDetector()
                if truth is None and frame.t - last_proc < min_dt * 0.95:
                    continue                      # faster than the target rate: skip this frame
                last_proc = frame.t
                self._process(frame, truth)
                self._report_status()
        finally:
            if self.source.ended and self._last_img is not None:
                last = Frame(self._last_img, self.engine.scene.t, time.time(), -1)
                if self.fuser is None:
                    for d in self.engine.flush():      # stream ended mid-encounter: judge what was seen
                        self._handle_decision(d, last, self._last_img)
                else:
                    self._fuse(self.engine.flush(), last, self._last_img, final=True)
            if self.learned is not None:
                self.learned.stop()
            self.recorder.flush()
            self._drain_clips(wait=True)
            self.source.stop()
            self._report_status(force=True, final=True)
            log.info("[%s] stopped", self.id)

    def _process(self, frame: Frame, truth: FrameObservation | None) -> None:
        t0 = time.perf_counter()
        img = frame.image
        if truth is not None:                     # synthetic camera: tracks come from the script
            mode, enhanced, dets = truth.lighting, img, truth.detections
            self.engine.ingest(truth)
            decisions = self.engine.evaluate_frame()
        else:
            res = self.pipeline.process(img, frame.t)
            mode, enhanced, dets, decisions = res.mode, res.enhanced, res.detections, res.decisions
            self.stats["det_ms"] = self.pipeline.perception.det_ms
            self.stats["pose_ms"] = self.pipeline.perception.pose_ms
        self._mode = mode

        # clips are labelled with the rule engine's (stitched) track ids
        canon = self.engine.scene.canonical
        dets = [Detection(canon(d.track_id), d.cls, d.box, d.conf) for d in dets]
        self.recorder.push(enhanced, frame.t, dets)
        self._last_img = enhanced
        if self.fuser is None:
            for d in decisions:
                self._handle_decision(d, frame, enhanced)
        else:
            if frame.t - self._last_offer >= self._learned_dt * 0.95:
                self._last_offer = frame.t
                self.learned.offer(img, frame.t)
            self._fuse(decisions, frame, enhanced)

        raised, cleared = (set(), set()) if truth is not None else self.tamper.update(img, frame.t, mode.value)
        for cond in raised:
            self._queue_tamper(cond, frame, active=True)
        for cond in cleared:
            self._queue_tamper(cond, frame, active=False)

        self._drain_clips()
        now = time.monotonic()
        self._fps_window = [x for x in self._fps_window if now - x < 5.0] + [now]
        span = self._fps_window[-1] - self._fps_window[0]
        self.stats["fps_proc"] = (len(self._fps_window) - 1) / span if span > 0 else 0.0
        self.stats["frames"] += 1
        self.stats["loop_ms"] = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    def _handle_decision(self, d: Decision, frame: Frame, img: np.ndarray) -> None:
        if d.level is Level.DISMISSED:
            self.stats["dismissed"] += 1
            log.info("[%s] dismissed encounter victim=%d bike=%d score=%.2f penalties=%s",
                     self.id, d.victim_id, d.bike_id, d.verdict.score, d.verdict.penalties)
            return
        self.stats["alerts" if d.level is Level.CONFIRMED else "suspects"] += 1
        event_id = str(uuid.uuid4())
        wall_now = frame.wall
        contact_wall = wall_now - (frame.t - d.verdict.contact_start)
        title = f"{d.level.value}  score {d.verdict.score:.2f}  {self.name}"
        roles = {d.victim_id: "victim", d.bike_id: "attacker" if d.mode == "on_foot" else "vehicle"}
        roles.update({rid: f"rider{i + 1}" for i, rid in enumerate(d.rider_ids)})
        mid_contact = (d.verdict.contact_start + d.verdict.contact_end) / 2
        snap = self.recorder.snapshot(event_id, mid_contact, roles, title, fallback=img)
        self.recorder.schedule_clip(event_id, d.verdict.contact_start, d.t, roles, title)
        payload = {
            "event_id": event_id,
            "type": "snatch",
            "node_id": self.node.get("node_id"),
            "site": self.node.get("site"),
            "camera_id": self.id,
            "camera_name": self.name,
            "occurred_at": _iso(contact_wall),
            "detected_at": _iso(wall_now),
            "video_time_s": round(d.verdict.contact_start, 2),
            **d.to_payload(),
            "source": "rules",
        }
        self.outbox.put("event", payload, {"snapshot": str(snap)})
        log.warning("[%s] %s score=%.2f victim=%d bike=%d latency=%.1fs", self.id, d.level.value,
                    d.verdict.score, d.victim_id, d.bike_id, d.t - d.verdict.contact_start)

    def _fuse(self, decisions: list[Decision], frame: Frame, img: np.ndarray, final: bool = False) -> None:
        self.fuser.add_windows(self.learned.drain())
        for d in decisions:
            if d.level is Level.DISMISSED:
                self.stats["dismissed"] += 1
                log.info("[%s] dismissed encounter victim=%d bike=%d score=%.2f penalties=%s",
                         self.id, d.victim_id, d.bike_id, d.verdict.score, d.verdict.penalties)
            self.fuser.add_rule(d.to_payload(), ref=d)
        for a in (self.fuser.flush() if final else self.fuser.poll(frame.t)):
            self._publish_fused(a, frame, img)

    def _publish_fused(self, a, frame: Frame, img: np.ndarray) -> None:
        """One fused alert (rule decision gated/upgraded/rescued by the model, or model-only) -> outbox."""
        pl, d = a.payload, a.ref
        level = pl["level"]
        self.stats["alerts" if level == "CONFIRMED" else "suspects"] += 1
        event_id = str(uuid.uuid4())
        cs, ce = pl["contact_start"], pl["contact_end"]
        contact_wall = frame.wall - (frame.t - cs)
        ms = pl.get("model_score")
        model_txt = f"  model {ms:.2f}" if ms is not None else ""
        if d is not None:
            roles = {d.victim_id: "victim", d.bike_id: "attacker" if d.mode == "on_foot" else "vehicle"}
            roles.update({rid: f"rider{i + 1}" for i, rid in enumerate(d.rider_ids)})
            base = d.to_payload()
            title = f"{level}  score {d.verdict.score:.2f}{model_txt}  {self.name}"
        else:
            roles = {}
            base = {"mode": "unknown", "score": pl["score"], "confidence": "low", "lighting": self._mode.value,
                    "contact_start": cs, "contact_end": ce, "decided_at": pl["decided_at"],
                    "latency_s": pl["latency_s"], "components": {}, "penalties": {}, "evidence": pl["evidence"],
                    "boxes": {}}
            title = f"{level} (learned scorer){model_txt}  {self.name}"
        snap = self.recorder.snapshot(event_id, (cs + ce) / 2, roles, title, fallback=img)
        self.recorder.schedule_clip(event_id, cs, frame.t, roles, title)
        payload = {
            "event_id": event_id, "type": "snatch", "node_id": self.node.get("node_id"),
            "site": self.node.get("site"), "camera_id": self.id, "camera_name": self.name,
            "occurred_at": _iso(contact_wall), "detected_at": _iso(frame.wall), "video_time_s": round(cs, 2),
            **base,
            "level": level, "source": pl["source"], "model_score": ms, "rule_score": pl.get("rule_score"),
            "rule_level": pl.get("rule_level"),
        }
        if pl.get("upgrade_of") in self._fusion_events:
            payload["supersedes"] = self._fusion_events[pl["upgrade_of"]]
        self._fusion_events[pl["fusion_id"]] = event_id
        self.outbox.put("event", payload, {"snapshot": str(snap)})
        log.warning("[%s] %s (%s) rule=%s model=%s latency=%.1fs", self.id, level, pl["source"],
                    pl.get("rule_score"), ms, frame.t - cs)

    def _queue_tamper(self, cond: str, frame: Frame, active: bool) -> None:
        event_id = str(uuid.uuid4())
        snap = self.recorder.dir / f"{event_id}_snapshot.jpg"
        cv2.imwrite(str(snap), frame.image)
        payload = {"event_id": event_id, "type": "tamper", "node_id": self.node.get("node_id"),
                   "site": self.node.get("site"), "camera_id": self.id, "camera_name": self.name,
                   "occurred_at": _iso(frame.wall), "detected_at": _iso(frame.wall),
                   "level": "TAMPER" if active else "TAMPER_CLEARED", "condition": cond,
                   "brightness": round(self.tamper.state.brightness, 1),
                   "sharpness": round(self.tamper.state.sharpness, 1),
                   "scene_match": round(self.tamper.state.scene_match, 3)}
        self.outbox.put("event", payload, {"snapshot": str(snap)})
        log.warning("[%s] tamper %s %s", self.id, cond, "raised" if active else "cleared")

    def _drain_clips(self, wait: bool = False) -> None:
        deadline = time.monotonic() + 30.0
        while True:
            for event_id, path in self.recorder.finished():
                if path is not None:
                    self.outbox.put("clip", {"event_id": event_id}, {"clip": str(path)})
            if not wait or not self.recorder.busy() or time.monotonic() > deadline:
                return
            time.sleep(0.2)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        thumb = None
        if self._last_img is not None:
            small = cv2.resize(self._last_img, (320, int(320 * self._last_img.shape[0] / self._last_img.shape[1])))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            thumb = base64.b64encode(jpg.tobytes()).decode() if ok else None
        fps = self.stats["fps_proc"]
        return {
            "id": self.id, "name": self.name, "status": self.source.status,
            "fps_in": round(self.source.fps_in, 1), "fps_proc": round(fps, 1),
            "degraded": self.stats["frames"] > 20 and fps < self.min_fps,
            "lighting": self._mode.value,
            "tamper": sorted(self.tamper.state.active),
            "tracks": sum(1 for tr in self.engine.scene.tracks.values() if tr.lost_at is None),
            "pairs": len(self.engine.pairs),
            "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.stats.items()},
            "thumbnail": thumb,
            **self._learned_status(),
        }

    def _learned_status(self) -> dict:
        lr = self.learned
        if lr is None:
            return {"learned": self.learned_state, "risk": None}
        state = "error" if lr.error else "on"
        return {"learned": state, "risk": None if lr.risk is None else round(lr.risk, 3),
                "learned_stats": {"scored": lr.scored, "dropped": lr.dropped}}

    def _report_status(self, force: bool = False, final: bool = False) -> None:
        if self.status_queue is None:
            return
        now = time.monotonic()
        if self.learned is not None and now - self._last_risk_push >= 0.5:
            self._last_risk_push = now           # small, frequent: the dashboard's live risk bar
            try:
                self.status_queue.put_nowait({"id": self.id, "kind": "risk", **self._learned_status()})
            except Exception:
                pass
        if not force and now - self._last_status < 2.0:
            return
        self._last_status = now
        st = self.status()
        if final:
            st["status"] = "ended" if self.source.ended else "stopped"
        try:
            self.status_queue.put_nowait(st)
        except Exception:
            pass

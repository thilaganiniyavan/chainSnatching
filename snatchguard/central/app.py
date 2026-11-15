"""Central alert server + dashboard.

Edge nodes POST events/clips/heartbeats (API key). The dashboard gets
everything live over Server-Sent Events. A watchdog marks cameras offline when
heartbeats stop; connection status stays in the camera network rather than the
incident feed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from snatchguard.central.db import Store, now_iso
from snatchguard.central.notify import TelegramNotifier
from snatchguard.central.footage import FootageRunner
from snatchguard.central.browsercam import BrowserCameras
from snatchguard.central import review as ai_review
from snatchguard.central import trim

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
OFFLINE_AFTER_S = 35.0
RECORDED_SITES = {"Uploaded footage", "Scripted demonstration", "Real CCTV demonstration · training clip"}


class Broadcaster:
    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()

    def publish(self, kind: str, data: dict) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        for q in list(self.clients):
            if q.qsize() < 500:
                q.put_nowait(msg)


class Review(BaseModel):
    status: str
    note: str | None = None


class LiveCamera(BaseModel):
    source: str = "0"                 # the server's webcam: a laptop camera standing in for CCTV
    name: str = "Live CCTV"
    model: str | None = None          # None: the installed default model
    sensitivity: str = "high"


class EventLink(BaseModel):
    event_id: str = Field(min_length=1, max_length=80)
    relation: str = "related_incident"
    note: str = Field(default="", max_length=1000)


class BrowserLive(BaseModel):
    name: str = Field(default="Live CCTV", max_length=100)
    site: str = Field(default="", max_length=100)


def create_app(data_dir: str | Path | None = None, api_keys: set[str] | None = None,
               dashboard_password: str | None = None, notifier=None, reviewer=None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("SNATCHGUARD_DATA", "runtime/central"))
    media = data_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    keys = api_keys or set(filter(None, os.environ.get("SNATCHGUARD_API_KEYS", "dev-edge-key").split(",")))
    store = Store(data_dir / "central.db")
    bus = Broadcaster()
    runner = FootageRunner(data_dir / "footage", next(iter(keys)))
    thumbnails: dict[str, bytes] = {}
    camera_origins = {origin.strip().rstrip("/") for origin in
                      os.environ.get("SNATCHGUARD_CAMERA_ORIGINS", "").split(",") if origin.strip()}
    browser_cameras = BrowserCameras(data_dir / "browser-cameras", next(iter(keys)),
                                    max(1, int(os.environ.get("SNATCHGUARD_MAX_BROWSER_CAMERAS", "4"))))

    def source_for(node_id):
        session = browser_cameras.for_node(node_id)
        return session.runner.source_for(node_id) if session else runner.source_for(node_id)

    def network_cameras():
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        return [camera for camera in store.cameras()
                if camera.get("site") not in RECORDED_SITES and camera["node_id"] not in {"edge-sim-01", "recorded-examples"}
                and camera["last_seen"] >= cutoff]

    live_risk: dict[str, dict] = {}          # node/camera -> latest learned-scorer risk (memory only)
    phone = notifier if notifier is not None else TelegramNotifier.from_env(data_dir)   # None: no phone alerts
    # AI review (snatchguard/central/review.py): snatch alerts are held until a vision model has looked at the clip.
    if reviewer is None and ai_review.enabled():
        reviewer = ai_review.AIReviewer()
    reviewing: set[str] = set()
    REVIEW_WAIT_S = float(os.environ.get("SNATCHGUARD_REVIEW_WAIT_S", "30"))
    MERGE_S = 20.0                          # reviewed alerts this close on one camera are one incident

    def is_hidden(ev: dict | None) -> bool:
        return ev is None or (ev["payload"].get("ai_review") or {}).get("state") in ("pending", "rejected", "merged")

    def same_incident(ev: dict) -> dict | None:
        """A snatch already shown for this camera within MERGE_S of this one (the same incident)."""
        t = ev["payload"].get("video_time_s")
        for other in store.list_events(50, "snatch", node_id=ev["node_id"]):
            if other["id"] == ev["id"] or other["camera_id"] != ev["camera_id"]:
                continue
            ot = other["payload"].get("video_time_s")
            if t is not None and ot is not None and abs(t - ot) <= MERGE_S:
                return other
        return None

    async def run_review(event_id: str, path: Path, at: float | None = None, result: dict | None = None) -> None:
        if event_id in reviewing:
            return
        reviewing.add(event_id)
        try:
            ev = store.get_event(event_id)
            if ev is None or (ev["payload"].get("ai_review") or {}).get("state") != "pending":
                return
            if at is None:
                at = ev["payload"].get("clip_event_at_s")
            if result is None:
                result = await asyncio.to_thread(reviewer.review, path, at)
            store.set_ai_review(event_id, result)
            log.info("AI review %s: %s %s", event_id, result.get("state"), result.get("reason") or result.get("error") or "")
            if result["state"] == "rejected":
                return
            if result["state"] == "confirmed":
                store.set_level(event_id, "CONFIRMED")
            ev = store.get_event(event_id)
            if same_incident(ev) is not None:
                store.set_ai_review(event_id, {**result, "state": "merged"})
                return
            bus.publish("event", ev)
            snap = media / event_id / "snapshot.jpg"
            notify(event_id, snap if snap.exists() else None)
            if phone is not None and ev.get("clip"):
                phone.clip(event_id, media / event_id / Path(ev["clip"]).name)
        finally:
            reviewing.discard(event_id)

    trimming: set[str] = set()

    BOX_WAIT_S = float(os.environ.get("SNATCHGUARD_BOX_WAIT_S", "45"))

    async def make_trimmed(ev: dict) -> dict | None:
        """Cut the detection as a proper video: from the uploaded file, or from the browser camera's frames.
        Returns {path, at (seconds into the video where the contact is), times (each frame's time), key (the clock
        the edge's boxes use), contact}, or None when the source is not available."""
        src = source_for(ev.get("node_id"))
        if src is None:
            return None
        out = event_dir(ev["id"]) / "trimmed.mp4"
        if src["kind"] == "file":
            t = float(ev["payload"].get("video_time_s") or 0.0)
            start = max(0.0, t - trim.PRE_S)
            await asyncio.to_thread(trim.trim_file, Path(src["path"]), start, t + trim.POST_S, out)
            import cv2
            cap = cv2.VideoCapture(str(out))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return {"path": out, "at": t - start, "times": [start + i / fps for i in range(n)], "key": "t",
                    "contact": t}
        t = datetime.fromisoformat(ev["occurred_at"]).timestamp()
        await asyncio.sleep(max(0.0, t + trim.POST_S + 0.5 - time.time()))        # let the rest happen first
        session = browser_cameras.for_node(ev.get("node_id"))
        if session is None:
            return None
        with session.cv:
            frames = [f for f in session.frames if t - trim.PRE_S <= f[0] <= t + trim.POST_S]
        if not frames:
            return None
        await asyncio.to_thread(trim.frames_to_mp4, frames, out)
        return {"path": out, "at": t - frames[0][0], "times": [f[0] for f in frames], "key": "wall", "contact": t}

    async def boxed(event_id: str, made: dict) -> Path | None:
        """The trimmed video with the detection boxes drawn, once the edge has sent its tracked boxes."""
        boxes = media / event_id / "boxes.json"
        deadline = time.time() + BOX_WAIT_S
        while not boxes.exists() and time.time() < deadline:
            await asyncio.sleep(0.5)
        if not boxes.exists():
            log.info("no boxes for %s; showing the plain trimmed video", event_id)
            return None
        try:
            data = json.loads(boxes.read_text())
            out = media / event_id / "trimmed_boxes.mp4"
            return await asyncio.to_thread(trim.annotate, made["path"], made["times"], data, made["contact"], out,
                                           made["key"])
        except Exception as exc:  # noqa: BLE001 - boxes are optional
            log.warning("boxes on the trimmed video for %s failed: %s", event_id, exc)
            return None

    async def handle_snatch(event_id: str) -> None:
        trimming.add(event_id)
        try:
            try:
                made = await make_trimmed(store.get_event(event_id))
            except Exception as exc:  # noqa: BLE001 - fall back to the edge's own clip
                log.warning("trimmed video for %s failed: %s", event_id, exc)
                made = None
            if made is not None:
                pending = (store.get_event(event_id)["payload"].get("ai_review") or {}).get("state") == "pending"
                # the AI reviews the clean video while the boxes are drawn for the operator
                review_task = (asyncio.create_task(asyncio.to_thread(reviewer.review, made["path"], made["at"]))
                               if pending and reviewer is not None else None)
                shown = await boxed(event_id, made) or made["path"]
                store.set_clip(event_id, f"/media/{event_id}/{shown.name}")
        finally:
            trimming.discard(event_id)
        if made is None:                         # use the edge's own clip if it already arrived
            edge = next(iter(sorted((media / event_id).glob("clip.*"))), None)
            ev = store.get_event(event_id)
            if edge is None:
                if reviewer is not None:
                    await review_fallback(event_id)  # waits for the edge clip (post_clip starts the review)
                return
            store.set_clip(event_id, f"/media/{event_id}/{edge.name}")
            if (ev["payload"].get("ai_review") or {}).get("state") == "pending":
                await run_review(event_id, edge, ev["payload"].get("clip_event_at_s"))
            return
        ev = store.get_event(event_id)
        if review_task is not None:
            await run_review(event_id, shown, made["at"], result=await review_task)
        elif not is_hidden(ev):
            bus.publish("event_update", ev)
            if phone is not None:
                phone.clip(event_id, shown)

    async def review_fallback(event_id: str) -> None:
        """No clip arrived in time: review the snapshot instead (or publish unreviewed if there is none)."""
        await asyncio.sleep(REVIEW_WAIT_S)
        ev = store.get_event(event_id)
        if ev is None or ev.get("clip") or event_id in reviewing or event_id in trimming or not is_hidden(ev):
            return
        snap = media / event_id / "snapshot.jpg"
        if snap.exists():
            await run_review(event_id, snap)
        elif (ev["payload"].get("ai_review") or {}).get("state") == "pending":
            store.set_ai_review(event_id, {"state": "unavailable", "error": "no media"})
            bus.publish("event", store.get_event(event_id))
            notify(event_id)

    def notify(event_id: str, snapshot: Path | None = None) -> None:
        if phone is not None:
            ev = store.get_event(event_id)
            phone.event({**ev["payload"], "level": ev["level"], "received_at": ev["received_at"]}, snapshot)

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(5)
            for session in browser_cameras.expire():
                mark_camera_offline(session)
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=OFFLINE_AFTER_S)
            for node in store.nodes():
                if node.get("site") in RECORDED_SITES or node["node_id"] in {"edge-sim-01", "recorded-examples"}:
                    continue
                if node["offline"] or datetime.fromisoformat(node["last_seen"]) >= cutoff:
                    continue
                store.mark_node_offline(node["node_id"])
                bus.publish("cameras", {"cameras": network_cameras()})
                log.warning("node %s went offline", node["node_id"])

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(watchdog())
        try:
            yield
        finally:
            task.cancel()
            browser_cameras.close()
            runner.stop()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if phone is not None:
            phone.close()

    app = FastAPI(title="SnatchGuard Central", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner
    app.state.browser_cameras = browser_cameras

    # Optional password for the dashboard and its read/review APIs (edge nodes use their API key instead).
    password = dashboard_password if dashboard_password is not None else os.environ.get("SNATCHGUARD_DASHBOARD_PASSWORD")
    user = os.environ.get("SNATCHGUARD_DASHBOARD_USER", "operator")
    if password:
        @app.middleware("http")
        async def dashboard_auth(request: Request, call_next):
            if (request.url.path == "/healthz" or request.headers.get("x-api-key") in keys or
                    (request.url.path == "/api/v1/browsercam/stream.mjpg" and request.query_params.get("key") in keys)):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("basic "):
                try:
                    u, _, pw = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
                except (ValueError, UnicodeDecodeError):
                    u, pw = "", ""
                if secrets.compare_digest(u, user) and secrets.compare_digest(pw, password):
                    return await call_next(request)
            from fastapi.responses import Response
            return Response("authentication required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="SnatchGuard"'})

    def local_port(request: Request) -> int:
        """The port this server listens on (the edge connects here directly, not through a tunnel's public host)."""
        server = request.scope.get("server")
        return int(os.environ.get("SNATCHGUARD_INTERNAL_PORT") or (server[1] if server and server[1] else 8000))

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key not in keys:
            raise HTTPException(401, "invalid API key")

    def event_dir(event_id: str) -> Path:
        if not SAFE_ID.match(event_id):
            raise HTTPException(400, "bad event id")
        d = media / event_id
        d.mkdir(exist_ok=True)
        return d

    # ---- edge-facing -------------------------------------------------------
    @app.post("/api/v1/events", dependencies=[Depends(require_key)])
    async def post_event(meta: str = Form(...), snapshot: UploadFile | None = File(default=None)):
        try:
            data = json.loads(meta)
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"meta is not JSON: {exc}") from exc
        if "event_id" not in data:
            raise HTTPException(422, "event_id missing")
        snap_url, path = None, None
        if snapshot is not None:
            path = event_dir(data["event_id"]) / "snapshot.jpg"
            path.write_bytes(await snapshot.read())
            snap_url = f"/media/{data['event_id']}/snapshot.jpg"
        held = reviewer is not None and data.get("type", "snatch") == "snatch"
        if held:
            data["ai_review"] = {"state": "pending"}
        created = store.insert_event(data, snap_url)
        if created and data.get("type", "snatch") == "snatch":
            bus.publish("detection", {"id": data["event_id"], "level": data.get("level", "SUSPECT"),
                                      "type": "snatch", "node_id": data.get("node_id"), "site": data.get("site"),
                                      "recorded_example": data.get("recorded_example", False)})
        if created and data.get("type", "snatch") == "snatch" and source_for(data.get("node_id")):
            asyncio.create_task(handle_snatch(data["event_id"]))
            if not held:
                bus.publish("event", store.get_event(data["event_id"]))
                notify(data["event_id"], path)
        elif created and held:
            asyncio.create_task(review_fallback(data["event_id"]))
        elif created:
            bus.publish("event", store.get_event(data["event_id"]))
            notify(data["event_id"], path)
            log.info("event %s %s from %s/%s", data.get("type"), data.get("level"), data.get("node_id"), data.get("camera_id"))
        return {"ok": True, "duplicate": not created}

    @app.post("/api/v1/events/{event_id}/clip", dependencies=[Depends(require_key)])
    async def post_clip(event_id: str, clip: UploadFile = File(...), boxes: UploadFile | None = File(default=None)):
        ext = Path(clip.filename or "clip.webm").suffix.lower()
        if ext not in (".webm", ".mp4", ".avi"):
            raise HTTPException(422, "unsupported clip type")
        if store.get_event(event_id) is None:
            raise HTTPException(404, "unknown event")      # edge retries; the event itself is sent first
        if boxes is not None:                    # the edge's tracked boxes, drawn on the trimmed detected video
            (event_dir(event_id) / "boxes.json").write_bytes((await boxes.read())[:5_000_000])
        path = event_dir(event_id) / f"clip{ext}"
        path.write_bytes(await clip.read())
        current = store.get_event(event_id)
        if event_id in trimming or "/trimmed" in (current.get("clip") or ""):
            return {"ok": True}                  # the trimmed detected video is used instead of the edge's clip
        store.set_clip(event_id, f"/media/{event_id}/clip{ext}")
        ev = store.get_event(event_id)
        if (ev["payload"].get("ai_review") or {}).get("state") == "pending":
            asyncio.create_task(run_review(event_id, path))
            return {"ok": True}
        if is_hidden(ev):
            return {"ok": True}
        bus.publish("event_update", ev)
        if phone is not None:
            phone.clip(event_id, path)
        return {"ok": True}

    @app.post("/api/v1/heartbeat", dependencies=[Depends(require_key)])
    async def heartbeat(request: Request):
        hb = await request.json()
        if "node_id" not in hb:
            raise HTTPException(422, "node_id missing")
        session = browser_cameras.for_node(hb["node_id"])
        if session and session.closed:
            return {"ok": True, "ignored": "camera disconnected"}
        for cam in hb.get("cameras", []):
            if cam.get("thumbnail"):
                with contextlib.suppress(ValueError):
                    thumbnails[f"{hb['node_id']}/{cam['id']}"] = base64.b64decode(cam["thumbnail"])
        store.upsert_heartbeat(hb)
        bus.publish("cameras", {"cameras": network_cameras(), "nodes": store.nodes()})
        return {"ok": True}

    @app.post("/api/v1/live", dependencies=[Depends(require_key)])
    async def post_live(request: Request):
        """Frequent, best-effort risk values (0..1) from each camera's learned scorer; not stored."""
        body = await request.json()
        node = body.get("node_id") if isinstance(body, dict) else None
        if not isinstance(node, str) or not SAFE_ID.match(node):
            raise HTTPException(422, "node_id missing or invalid")
        cams = body.get("cameras", [])
        if not isinstance(cams, list):
            raise HTTPException(422, "cameras must be a list")
        out = []
        for cam in cams[:64]:
            cid = cam.get("id") if isinstance(cam, dict) else None
            if not isinstance(cid, str) or not SAFE_ID.match(cid):
                continue
            risk = cam.get("risk")
            if isinstance(risk, (int, float)) and not isinstance(risk, bool):
                risk = round(min(1.0, max(0.0, float(risk))), 3)
            else:
                risk = None
            state = cam.get("learned") if cam.get("learned") in ("on", "off", "missing", "error") else None
            entry = {"node_id": node, "camera_id": cid, "risk": risk, "learned": state, "at": now_iso()}
            live_risk[f"{node}/{cid}"] = entry
            out.append(entry)
        if out:
            bus.publish("risk", {"cameras": out})
        return {"ok": True, "cameras": len(out)}

    @app.get("/api/v1/live")
    def get_live():
        return {"cameras": list(live_risk.values())}

    # One local processing job at a time; uploads are kept outside the public media directory.
    @app.get("/api/v1/footage")
    def footage_status():
        return runner.status()

    @app.get("/api/v1/footage/models")
    def footage_models():
        return {"models": runner.models()}

    @app.post("/api/v1/footage/live")
    def footage_live(camera: LiveCamera, request: Request):
        source = camera.source.strip()
        try:
            parsed = urlsplit(source)
            valid = source.isascii() and source.isdigit() and len(source) <= 3
            valid = valid or (parsed.scheme in ("rtsp", "rtsps", "http", "https") and bool(parsed.hostname))
        except ValueError:
            valid = False
        if not valid or len(source) > 2048 or any(ord(c) < 32 for c in source):
            raise HTTPException(422, "Enter a webcam index or an RTSP/HTTP camera URL.")
        try:
            return runner.start(source, camera.name.strip()[:100] or "Live camera", local_port(request),
                                live=True, model=camera.model, sensitivity=camera.sensitivity)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/footage/real-demo")
    def footage_real_demo(request: Request):
        from snatchguard.core.config import REPO_ROOT
        clip = REPO_ROOT / "data/demo/real_cctv_demo.mp4"      # web CCTV clip OTGVGhOqJOo_0 (a training clip)
        if not clip.is_file():
            raise HTTPException(404, "The real demonstration clip is not available.")
        try:
            return runner.start(str(clip), "Real CCTV demo (training clip)", local_port(request), real_demo=True)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/footage/output")
    def footage_output(request: Request):
        """The current or last footage / live job's detected incidents: one 20 s evidence clip each. An alert that was
        upgraded (SUSPECT -> CONFIRMED) is replaced by its upgrade, so every incident appears once."""
        selected = session_owner(request).runner if request.headers.get("x-camera-id") else runner
        job = selected.status()
        if not job.get("id"):
            return {"job": job, "incidents": []}
        events = [e for e in store.list_events(500, "snatch", node_id="footage-" + job["id"])]
        replaced = {e["payload"].get("supersedes") for e in events}
        incidents = sorted((e for e in events if e["id"] not in replaced), key=lambda e: e["payload"].get("video_time_s") or 0)
        return {"job": job, "incidents": [
            {"id": e["id"], "level": e["level"], "score": e["score"], "video_time_s": e["payload"].get("video_time_s"),
             "ai_verified": (e["payload"].get("ai_review") or {}).get("state") == "confirmed",
             "clip_removed": bool(e["payload"].get("clip_removed")),
             "clip_event_at_s": e["payload"].get("clip_event_at_s"),
             "occurred_at": e["occurred_at"], "clip": e["clip"], "snapshot": e["snapshot"]} for e in incidents]}

    def session_owner(request: Request):
        session = browser_cameras.owned(request.headers.get("x-camera-id", ""),
                                        request.headers.get("x-camera-token", ""))
        if session is None:
            raise HTTPException(403, "This camera session belongs to another connection or has expired.")
        return session

    def mark_camera_offline(session):
        if session.runner.job:
            store.mark_node_offline(session.runner.job["node_id"])
            bus.publish("cameras", {"cameras": network_cameras(), "nodes": store.nodes()})

    @app.get("/api/v1/browsercam/token")
    def browsercam_token():
        try:
            session = browser_cameras.create()
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"id": session.id, "token": session.token}

    @app.websocket("/api/v1/browsercam/ws")
    async def browsercam_ws(ws: WebSocket):
        session = browser_cameras.owned(ws.query_params.get("id", ""), ws.query_params.get("token", ""))
        if session is None:
            await ws.close(code=1008)
            return
        origin = ws.headers.get("origin")
        if origin and origin.rstrip("/") not in camera_origins and urlsplit(origin).netloc != ws.headers.get("host"):
            await ws.close(code=1008)
            return
        with session.cv:
            if session.closed or session.connected or time.time() - session.created > 60:
                await ws.close(code=1008)
                return
            session.connected = True
        await ws.accept()
        try:
            while not session.closed:
                data = await asyncio.wait_for(ws.receive_bytes(), timeout=15)
                if not 100 < len(data) <= 500_000 or data[:2] != b"\xff\xd8":
                    await ws.close(code=1009)
                    break
                session.push(data)
                if session.seq == 1:
                    await ws.send_json({"ready": True})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            session.close()
            mark_camera_offline(session)
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await ws.close()

    @app.get("/api/v1/browsercam/stream.mjpg")
    def browsercam_stream(id: str = "", key: str = ""):
        if key not in keys:
            raise HTTPException(401, "invalid key")
        session = browser_cameras.sessions.get(id)
        if session is None or session.closed:
            raise HTTPException(404, "camera unavailable")

        def frames():
            seen = 0
            while not session.closed:
                with session.cv:
                    session.cv.wait_for(lambda: session.seq != seen or session.closed, timeout=5.0)
                    if session.closed:
                        break
                    if session.seq == seen:
                        continue
                    jpg, seen = session.jpg, session.seq
                if jpg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() +
                           b"\r\n\r\n" + jpg + b"\r\n")

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/v1/footage/browser-live")
    def footage_browser_live(request: Request, camera: BrowserLive):
        session = session_owner(request)
        with session.cv:
            if session.closed or not session.connected or not session.jpg or time.time() - session.at > 15:
                raise HTTPException(409, "Connect this device's camera before starting detection.")
        port = local_port(request)
        internal = os.environ.get("SNATCHGUARD_INTERNAL_URL", f"http://127.0.0.1:{port}").rstrip("/")
        from urllib.parse import urlencode
        source = internal + "/api/v1/browsercam/stream.mjpg?" + urlencode({"key": next(iter(keys)), "id": session.id})
        try:
            job = session.start(source, camera.name.strip() or "Live CCTV", port,
                                camera.site.strip() or "Live camera")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        store.upsert_heartbeat({"node_id": job["node_id"], "site": camera.site.strip() or "Live camera",
                                "cameras": [{"id": job["camera_id"], "name": camera.name.strip() or "Live CCTV",
                                             "status": "starting"}]})
        thumbnails[f"{job['node_id']}/{job['camera_id']}"] = session.jpg
        bus.publish("cameras", {"cameras": network_cameras(), "nodes": store.nodes()})
        return {k: v for k, v in job.items() if k != "message"}

    @app.get("/api/v1/browsercam/status")
    def browsercam_status(request: Request):
        session = session_owner(request)
        return {**session.runner.status(), "connected": session.connected and not session.closed}

    @app.post("/api/v1/browsercam/stop")
    def browsercam_stop(request: Request):
        session = session_owner(request)
        session.close()
        mark_camera_offline(session)
        return session.runner.status()

    @app.post("/api/v1/footage/stop")
    def footage_stop():
        return runner.stop()

    @app.post("/api/v1/footage/upload")
    async def footage_upload(request: Request, video: UploadFile = File(...), model: str | None = Form(None),
                            sensitivity: str = Form("high")):
        if runner.status()["state"] == "running":
            raise HTTPException(409, "A video is already being processed.")
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            raise HTTPException(422, "Choose an MP4, AVI, MOV, MKV or WebM video.")
        import uuid
        path = runner.directory / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with path.open("wb") as out:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > 500 * 1024 * 1024:
                        raise HTTPException(413, "Video exceeds the 500 MB limit.")
                    out.write(chunk)
            import cv2
            cap = cv2.VideoCapture(str(path))
            valid, _ = cap.read()
            cap.release()
            if not valid:
                raise HTTPException(422, "The video could not be decoded.")
            return runner.start(str(path), Path(video.filename or "Uploaded video").name[:100], local_port(request),
                                model=model or None, sensitivity=sensitivity)
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await video.close()

    # ---- dashboard-facing --------------------------------------------------
    @app.get("/api/v1/events")
    def list_events(limit: int = 100, type: str | None = None, status: str | None = None, level: str | None = None,
                    deleted: bool = False, include_deleted: bool = False):
        return store.list_events(min(limit, 500), type, status, level, deleted=None if include_deleted else deleted)

    @app.get("/api/v1/events/{event_id}")
    def get_event(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev):
            raise HTTPException(404, "not found")
        return ev

    @app.delete("/api/v1/events/{event_id}")
    def delete_alert(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev):
            raise HTTPException(404, "event not found")
        store.delete_alert(event_id)
        ev = store.get_event(event_id)
        bus.publish("event_update", ev)
        return ev

    @app.post("/api/v1/events/{event_id}/restore")
    def restore_alert(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev):
            raise HTTPException(404, "event not found")
        store.delete_alert(event_id, restore=True)
        ev = store.get_event(event_id)
        bus.publish("event_update", ev)
        return ev

    @app.delete("/api/v1/events/{event_id}/video")
    def remove_event_video(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev):
            raise HTTPException(404, "event not found")
        store.video_visibility(event_id, False)
        ev = store.get_event(event_id)
        bus.publish("event_update", ev)
        return ev

    @app.post("/api/v1/events/{event_id}/video/restore")
    def restore_event_video(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev):
            raise HTTPException(404, "event not found")
        store.video_visibility(event_id, True)
        ev = store.get_event(event_id)
        bus.publish("event_update", ev)
        return ev

    @app.get("/api/v1/events/{event_id}/related")
    def related_events(event_id: str):
        ev = store.get_event(event_id)
        if is_hidden(ev) or ev["type"] != "snatch":
            raise HTTPException(404, "snatch event not found")
        links = []
        linked_ids = set()
        for link in store.event_links(event_id):
            other = store.get_event(link["other_id"])
            if not is_hidden(other):
                links.append({**link, "event": other})
                linked_ids.add(other["id"])
        candidates = []
        site = (ev.get("site") or "").strip().casefold()
        # Generic locations do not establish camera proximity. Never compare camera-local track IDs.
        if site and site not in {"live camera", "uploaded footage", "scripted demonstration"}:
            for other in store.list_events(500, "snatch"):
                if other["id"] == event_id or other["id"] in linked_ids or other["status"] == "false_alarm":
                    continue
                if (other["node_id"], other["camera_id"]) == (ev["node_id"], ev["camera_id"]):
                    continue
                if (other.get("site") or "").strip().casefold() != site:
                    continue
                try:
                    delta = abs((datetime.fromisoformat(ev["occurred_at"]) -
                                 datetime.fromisoformat(other["occurred_at"])).total_seconds())
                except (TypeError, ValueError):
                    continue
                if delta <= 120:
                    candidates.append({"event": other, "seconds_apart": delta,
                                       "reason": "Same site, different camera, within two minutes"})
        candidates.sort(key=lambda c: c["seconds_apart"])
        return {"links": links, "candidates": candidates[:10],
                "basis": "Time and site only; these suggestions do not identify a person."}

    def valid_link(event_id: str, other_id: str):
        first, second = store.get_event(event_id), store.get_event(other_id)
        if is_hidden(first) or is_hidden(second):
            raise HTTPException(404, "event not found")
        if event_id == other_id or first["type"] != "snatch" or second["type"] != "snatch":
            raise HTTPException(422, "Choose two different snatch alerts.")
        if (first["node_id"], first["camera_id"]) == (second["node_id"], second["camera_id"]):
            raise HTTPException(422, "Choose an alert from a different camera.")

    @app.post("/api/v1/events/{event_id}/links")
    def link_event(event_id: str, body: EventLink):
        valid_link(event_id, body.event_id)
        if body.relation not in {"related_incident", "possible_same_person"}:
            raise HTTPException(422, "Choose a supported relationship.")
        store.link_events(event_id, body.event_id, body.relation, body.note)
        for eid in (event_id, body.event_id):
            bus.publish("event_update", store.get_event(eid))
        return related_events(event_id)

    @app.delete("/api/v1/events/{event_id}/links/{other_id}")
    def unlink_event(event_id: str, other_id: str):
        valid_link(event_id, other_id)
        store.unlink_events(event_id, other_id)
        for eid in (event_id, other_id):
            bus.publish("event_update", store.get_event(eid))
        return related_events(event_id)

    @app.post("/api/v1/events/{event_id}/review")
    def review(event_id: str, body: Review):
        if body.status not in ("acknowledged", "confirmed", "false_alarm", "new"):
            raise HTTPException(422, "bad status")
        if not store.review(event_id, body.status, body.note):
            raise HTTPException(404, "not found")
        ev = store.get_event(event_id)
        bus.publish("event_update", ev)
        return ev

    @app.get("/api/v1/cameras")
    def cameras():
        return {"cameras": network_cameras(), "nodes": store.nodes()}

    @app.get("/api/v1/cameras/{node_id}/{camera_id}/thumbnail.jpg")
    def thumbnail(node_id: str, camera_id: str):
        data = thumbnails.get(f"{node_id}/{camera_id}")
        if data is None:
            raise HTTPException(404, "no thumbnail yet")
        from fastapi.responses import Response
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/stats")
    def stats():
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        return store.stats(since, real_only=True)

    @app.get("/api/v1/stream")
    async def stream(request: Request):
        q: asyncio.Queue = asyncio.Queue()
        bus.clients.add(q)

        async def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        yield await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                bus.clients.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"ok": True, "time": now_iso()})

    @app.get("/metrics")
    def metrics_page():
        return FileResponse(STATIC / "metrics.html")

    @app.get("/footage")
    def footage_page():
        return FileResponse(STATIC / "footage.html")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/media", StaticFiles(directory=media), name="media")
    return app

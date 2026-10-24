"""Central alert server + dashboard.

Edge nodes POST events/clips/heartbeats (API key). The dashboard gets
everything live over Server-Sent Events. A watchdog raises an alert when a
node stops sending heartbeats, so a dead or disconnected camera is never
mistaken for a quiet street.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from snatchguard.central.db import Store, now_iso
from snatchguard.central.notify import TelegramNotifier

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
OFFLINE_AFTER_S = 35.0


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


def create_app(data_dir: str | Path | None = None, api_keys: set[str] | None = None,
               dashboard_password: str | None = None, notifier=None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("SNATCHGUARD_DATA", "runtime/central"))
    media = data_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    keys = api_keys or set(filter(None, os.environ.get("SNATCHGUARD_API_KEYS", "dev-edge-key").split(",")))
    store = Store(data_dir / "central.db")
    bus = Broadcaster()
    thumbnails: dict[str, bytes] = {}
    live_risk: dict[str, dict] = {}          # node/camera -> latest learned-scorer risk (memory only)
    phone = notifier if notifier is not None else TelegramNotifier.from_env(data_dir)   # None: no phone alerts

    def notify(event_id: str, snapshot: Path | None = None) -> None:
        if phone is not None:
            ev = store.get_event(event_id)
            phone.event({**ev["payload"], "level": ev["level"], "received_at": ev["received_at"]}, snapshot)

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(5)
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=OFFLINE_AFTER_S)
            for node in store.nodes():
                if node["offline"] or datetime.fromisoformat(node["last_seen"]) >= cutoff:
                    continue
                store.mark_node_offline(node["node_id"])
                ev = {"event_id": f"offline-{node['node_id']}-{int(datetime.now().timestamp())}", "type": "camera_offline",
                      "level": "OFFLINE", "node_id": node["node_id"], "site": node.get("site"),
                      "camera_name": f"Edge node {node['node_id']}", "occurred_at": node["last_seen"],
                      "detected_at": now_iso(),
                      "detail": f"No heartbeat for over {int(OFFLINE_AFTER_S)} s: cameras on this node are NOT monitored."}
                store.insert_event(ev, None)
                bus.publish("event", store.get_event(ev["event_id"]))
                notify(ev["event_id"])
                bus.publish("cameras", {"cameras": store.cameras()})
                log.warning("node %s went offline", node["node_id"])

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(watchdog())
        yield
        task.cancel()
        if phone is not None:
            phone.close()

    app = FastAPI(title="SnatchGuard Central", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    # Optional password for the dashboard and its read/review APIs (edge nodes use their API key instead).
    password = dashboard_password if dashboard_password is not None else os.environ.get("SNATCHGUARD_DASHBOARD_PASSWORD")
    user = os.environ.get("SNATCHGUARD_DASHBOARD_USER", "operator")
    if password:
        import secrets

        @app.middleware("http")
        async def dashboard_auth(request: Request, call_next):
            if request.url.path == "/healthz" or request.headers.get("x-api-key") in keys:
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
        created = store.insert_event(data, snap_url)
        if created:
            bus.publish("event", store.get_event(data["event_id"]))
            notify(data["event_id"], path)
            log.info("event %s %s from %s/%s", data.get("type"), data.get("level"), data.get("node_id"), data.get("camera_id"))
        return {"ok": True, "duplicate": not created}

    @app.post("/api/v1/events/{event_id}/clip", dependencies=[Depends(require_key)])
    async def post_clip(event_id: str, clip: UploadFile = File(...)):
        ext = Path(clip.filename or "clip.webm").suffix.lower()
        if ext not in (".webm", ".mp4", ".avi"):
            raise HTTPException(422, "unsupported clip type")
        if store.get_event(event_id) is None:
            raise HTTPException(404, "unknown event")      # edge retries; the event itself is sent first
        path = event_dir(event_id) / f"clip{ext}"
        path.write_bytes(await clip.read())
        store.set_clip(event_id, f"/media/{event_id}/clip{ext}")
        bus.publish("event_update", store.get_event(event_id))
        if phone is not None:
            phone.clip(event_id, path)
        return {"ok": True}

    @app.post("/api/v1/heartbeat", dependencies=[Depends(require_key)])
    async def heartbeat(request: Request):
        hb = await request.json()
        if "node_id" not in hb:
            raise HTTPException(422, "node_id missing")
        was_offline = any(n["node_id"] == hb["node_id"] and n["offline"] for n in store.nodes())
        for cam in hb.get("cameras", []):
            if cam.get("thumbnail"):
                with contextlib.suppress(ValueError):
                    thumbnails[f"{hb['node_id']}/{cam['id']}"] = base64.b64decode(cam["thumbnail"])
        store.upsert_heartbeat(hb)
        if was_offline:
            ev = {"event_id": f"online-{hb['node_id']}-{int(datetime.now().timestamp())}", "type": "camera_online",
                  "level": "ONLINE", "node_id": hb["node_id"], "site": hb.get("site"),
                  "camera_name": f"Edge node {hb['node_id']}", "occurred_at": now_iso(), "detected_at": now_iso(),
                  "detail": "Heartbeats resumed; monitoring restored."}
            store.insert_event(ev, None)
            bus.publish("event", store.get_event(ev["event_id"]))
        bus.publish("cameras", {"cameras": store.cameras(), "nodes": store.nodes()})
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

    # ---- dashboard-facing --------------------------------------------------
    @app.get("/api/v1/events")
    def list_events(limit: int = 100, type: str | None = None, status: str | None = None, level: str | None = None):
        return store.list_events(min(limit, 500), type, status, level)

    @app.get("/api/v1/events/{event_id}")
    def get_event(event_id: str):
        ev = store.get_event(event_id)
        if ev is None:
            raise HTTPException(404, "not found")
        return ev

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
        return {"cameras": store.cameras(), "nodes": store.nodes()}

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
        return store.stats(since)

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

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/media", StaticFiles(directory=media), name="media")
    return app

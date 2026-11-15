"use strict";
// Footage page: 01 live CCTV (the visitor's own laptop camera, streamed to the server), 02 upload a video.
// The server picks the model and settings.
const $ = (s) => document.querySelector(s);
const message = (s) => { $("#message").textContent = s; $("#message").hidden = !s; };
let job = { state: "idle" };
let cameraSession = null;
let startingCamera = false;
const cameraHeaders = () => cameraSession ? {"x-camera-id": cameraSession.id, "x-camera-token": cameraSession.token} : {};
let cam = null;           // {stream, ws, timer} while this tab is sending its camera

async function call(url, options) {
  const res = await fetch(url, options);
  let body = {};
  try { body = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) throw new Error(body.detail || "The request failed. Please try again.");
  return body;
}

function render() {
  const running = job.state === "running";
  const live = running && job.kind === "live camera";
  const upload = running && job.kind === "uploaded video";
  $("#start-live").hidden = live;
  $("#start-live").disabled = startingCamera || live;
  $("#stop-live").hidden = !live;
  $("#analyse").disabled = running;
  $("#stop-upload").hidden = !upload;
  $("#live-badge").hidden = !live;
  const ready = live && job.connected && job.camera_status === "online";
  $("#live-badge").textContent = ready ? "● LIVE · DETECTING" : "◌ STARTING DETECTOR… (about 30 s)";
  $("#live-badge").classList.toggle("starting", live && !ready);
  if (!live && !cam) { $("#cam-video").hidden = true; $("#live-placeholder").hidden = false; $("#live-placeholder").textContent = upload ? "Camera is off while a video is analysed" : "Camera is off"; }
  if (live && !cam) { $("#live-placeholder").hidden = false; $("#live-placeholder").textContent = "This camera connection is closing. Start live CCTV again once it has stopped."; }
  $("#upload-progress").hidden = !(upload || (job.kind === "uploaded video" && job.state !== "idle"));
  $("#upload-progress").classList.toggle("done", !upload);
  const sessionBar = $("#live-session-bar");
  if (sessionBar) {
    sessionBar.classList.toggle("hidden", !cam || !$(".shell").classList.contains("hidden"));
    $("#overview-camera-status").textContent = ready ? "This device · live detection active" : live ? "This device · detector starting" : "This device · camera connecting";
  }
  $("#upload-state").textContent = upload ? `Analysing ${job.name}…` : job.state === "completed" ? `Finished ${job.name}` : job.state === "failed" ? "The video could not be analysed." : job.state === "stopped" ? "Stopped." : "";
}

async function status() {
  try {
    job = await call(cameraSession ? "/api/v1/browsercam/status" : "/api/v1/footage", {headers: cameraHeaders()});
    if (cam && !startingCamera && ["failed", "completed", "stopped"].includes(job.state)) {
      stopCamera();
      fetch("/api/v1/browsercam/stop", {method: "POST", headers: cameraHeaders()}).catch(() => {});
      message("The detector stopped. Start live CCTV to reconnect this camera.");
    }
    if (job.state === "running" && job.kind === "live camera") {
      const { cameras } = await (await fetch("/api/v1/cameras")).json();
      job.camera_status = (cameras || []).find((c) => c.node_id === job.node_id)?.status || "starting";
    }
  } catch (_) { job = { state: "idle" }; }
  render();
}

function stopCamera() {
  if (!cam) return;
  clearInterval(cam.timer);
  try { cam.ws.close(); } catch (_) { /* closed */ }
  cam.stream.getTracks().forEach((t) => t.stop());
  cam = null;
  $("#cam-video").srcObject = null;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot open the camera here. Use Chrome, Edge or Safari over the https link.");
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false });
  const video = $("#cam-video");
  video.srcObject = stream; video.hidden = false; $("#live-placeholder").hidden = true;
  cam = { stream, ws: null, timer: null };
  await video.play();
  cameraSession = await call("/api/v1/browsercam/token");
  const { token, id } = cameraSession;
  const cameraOrigin = document.querySelector('meta[name="snatchguard-camera-origin"]')?.content || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
  const ws = new WebSocket(`${cameraOrigin}/api/v1/browsercam/ws?id=${encodeURIComponent(id)}&token=${encodeURIComponent(token)}`);
  cam.ws = ws;
  ws.binaryType = "arraybuffer";
  await new Promise((ok, fail) => {
    const timeout = setTimeout(() => { ws.close(); fail(new Error("Camera connection timed out.")); }, 10000);
    ws.onopen = () => { clearTimeout(timeout); ok(); };
    ws.onerror = ws.onclose = () => { clearTimeout(timeout); fail(new Error("Could not connect the camera to the server.")); };
  });
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  let encoding = false;
  const send = () => {                                   // about 10 frames a second, skipped when the link is slow
    if (encoding || ws.readyState !== 1 || ws.bufferedAmount > 400000 || !video.videoWidth) return;
    const w = Math.min(960, video.videoWidth);
    canvas.width = w; canvas.height = Math.round(w * video.videoHeight / video.videoWidth);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    encoding = true;
    canvas.toBlob(async (b) => {
      try { if (b && ws.readyState === 1) { const buf = await b.arrayBuffer(); if (ws.readyState === 1) ws.send(buf); } }
      finally { encoding = false; }
    }, "image/jpeg", 0.75);
  };
  cam = { stream, ws, timer: setInterval(send, 100) };
  const ready = new Promise((ok, fail) => {
    const timeout = setTimeout(() => fail(new Error("No camera frames reached the server.")), 10000);
    ws.onmessage = () => { clearTimeout(timeout); ok(); };
    ws.onclose = () => {
      clearTimeout(timeout);
      if (cam && cam.ws === ws) { stopCamera(); message("Camera disconnected. Start live CCTV to reconnect."); status(); }
      fail(new Error("Camera disconnected."));
    };
  });
  send();
  await ready;
}

$("#start-live").addEventListener("click", async () => {
  if (startingCamera || cam) return;
  startingCamera = true;
  message("");
  $("#start-live").disabled = true;
  try {
    await startCamera();
    job = await call("/api/v1/footage/browser-live", {
      method: "POST", headers: {...cameraHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({name: $("#camera-name").value, site: $("#camera-site").value})
    });
    render();
  } catch (err) {
    stopCamera();
    message(err.name === "NotAllowedError" ? "Camera permission was denied. Allow the camera for this site and try again." : err.message);
    await status();
  } finally { startingCamera = false; render(); }
});
const stop = async () => { try { job = await call(cameraSession ? "/api/v1/browsercam/stop" : "/api/v1/footage/stop", { method: "POST", headers: cameraHeaders() }); } catch (err) { message(err.message); } stopCamera(); render(); };
$("#stop-live").addEventListener("click", stop);
$("#stop-upload").addEventListener("click", stop);
$("#video").addEventListener("change", (e) => { $("#filename").textContent = e.target.files[0]?.name || "Choose a video file"; });
$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("#video").files[0];
  if (!file) { message("Choose a video file first."); return; }
  cameraSession = null;
  const form = new FormData();
  form.append("video", file);
  $("#analyse").disabled = true;
  $("#upload-progress").hidden = false;
  $("#upload-state").textContent = "Uploading…";
  try { job = await call("/api/v1/footage/upload", { method: "POST", body: form }); render(); }
  catch (err) { message(err.message); await status(); }
});

const legacyVideoState = new Map();
const VIDEO_REMOVED_NOTE = "[Deleted from portal]";
async function changeOutputVideo(incident, button) {
  button.disabled = true;
  try {
    const restore = incident.clip_removed;
    let res = await fetch(`/api/v1/events/${encodeURIComponent(incident.id)}${restore ? "/restore" : ""}`, {method: restore ? "POST" : "DELETE"});
    if ([404, 405, 422].includes(res.status)) {
      const event = await call(`/api/v1/events/${encodeURIComponent(incident.id)}`);
      let note = event.review_note || "";
      for (const marker of [VIDEO_REMOVED_NOTE, "[Dismissed from portal]", "[Video removed from alert]"]) note = note.split(marker).join("");
      note = note.trim();
      if (!restore) note = `${VIDEO_REMOVED_NOTE}${note ? " " + note : ""}`;
      await call(`/api/v1/events/${encodeURIComponent(incident.id)}/review`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({status: event.status || "new", note: note || null})
      });
    } else if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "The video could not be updated.");
    }
    legacyVideoState.set(incident.id, !restore);
    outputKey = "";
    await output();
  } catch (error) { message(error.message); button.disabled = false; }
}

let outputKey = "";
async function output() {
  try {
    const res = await fetch("/api/v1/footage/output", {headers: cameraHeaders()});
    if (!res.ok) return;
    const data = await res.json();
    const j = data.job || {};
    // A previous demonstration is not a detection from the visitor's camera or upload.
    let items = ["live camera", "uploaded video"].includes(j.kind) ? (data.incidents || []) : [];
    if (items.some(i => i.clip_removed === undefined)) {
      const events = await call("/api/v1/events?limit=500");
      const byId = new Map(events.map(event => [event.id, event]));
      items = items.map(incident => {
        const event = byId.get(incident.id);
        const removed = event ? !!(event.payload?.deleted || event.payload?.dismissed || event.payload?.clip_removed) || [VIDEO_REMOVED_NOTE, "[Dismissed from portal]", "[Video removed from alert]"].some(marker => (event.review_note || "").includes(marker)) : legacyVideoState.get(incident.id);
        if (event) legacyVideoState.set(incident.id, removed);
        return {...incident, clip_removed: !!removed, clip: removed ? null : incident.clip};
      });
    }
    items = items.filter(item => !item.clip_removed && !item.deleted);
    const key = (data.job?.id || "") + "|" + data.job?.state + "|" + items.map((i) => i.id + (i.clip || "") + (i.clip_removed ? "removed" : "")).join(",");
    if (key === outputKey) return;
    outputKey = key;
    const isLive = j.kind === "live camera";
    const running = j.state === "running";
    $("#output-summary").textContent = !j.id || !["live camera", "uploaded video"].includes(j.kind)
      ? "Start the live camera or upload a video. Each detection appears here with its trimmed detected video."
      : items.length ? `${items.length} snatch${items.length > 1 ? "es" : ""} detected ${isLive ? "on the live camera" : "in " + j.name}${running ? " so far" : ""}.`
      : running ? (isLive ? "Watching the live camera. No snatch detected yet." : `Analysing ${j.name}. No snatch detected yet.`)
      : `No snatch detected ${isLive ? "on the live camera" : "in " + j.name}.`;
    const list = $("#output-list");
    list.replaceChildren();
    items.forEach((i, n) => {
      const card = document.createElement("div");
      card.className = "output-card";
      const when = isLive ? new Date(i.occurred_at).toLocaleTimeString() : `${Number(i.video_time_s || 0).toFixed(1)} s into the video`;
      const head = document.createElement("div");
      head.className = "output-head";
      const level = document.createElement("span");
      level.className = "output-level" + (i.level === "CONFIRMED" ? " confirmed" : "");
      level.textContent = i.level === "CONFIRMED" ? "CONFIRMED" : "SUSPECTED";
      head.append(level);
      head.append(` Snatch ${n + 1} · ${when}`);
      card.append(head);
      if (i.clip) {
        const v = document.createElement("video");
        v.src = i.clip; v.controls = true; v.muted = true; v.playsInline = true; v.preload = "metadata";
        const a = document.createElement("a");
        a.href = i.clip;
        a.download = isLive ? `snatch_${n + 1}_${i.occurred_at.replace(/[:.]/g, "-")}.${(i.clip.split(".").pop() || "mp4")}` : `snatch_${n + 1}_${Math.round(i.video_time_s || 0)}s.${(i.clip.split(".").pop() || "mp4")}`;
        a.className = "action secondary clip-download"; a.textContent = "⇩ Download trimmed video";
        card.append(v, a);
      } else {
        const w = document.createElement("div");
        w.className = "clip-wait"; w.textContent = i.clip_removed ? "Video removed from this alert." : "Preparing the trimmed detected video…";
        card.append(w);
      }
      if (i.clip) {
        const button = document.createElement("button");
        button.className = "action secondary video-action";
        button.textContent = "🗑";
        const label = "Move alert to Deleted";
        button.title = label; button.setAttribute("aria-label", label);
        button.addEventListener("click", () => changeOutputVideo(i, button));
        card.append(button);
      }
      list.append(card);
    });
  } catch (_) { /* keep the last view */ }
}

// Keep this document (and its camera stream) alive when switching to Overview.
let overviewPage = null;
let overviewLoading = false;
const cameraShell = $(".shell");

function showFootage(push = true) {
  if (!overviewPage) return;
  overviewPage.classList.add("hidden");
  cameraShell.classList.remove("hidden");
  const video = $("#cam-video");
  $("#live-view").prepend(video);
  if (cam) video.play().catch(() => {});
  if (push) history.pushState({cameraView: "footage"}, "", "/footage");
  document.title = "Footage & demo | SnatchGuard";
  render();
}

async function showOverview(push = true) {
  if (overviewLoading) return;
  overviewLoading = true;
  try {
    if (!overviewPage) {
      const response = await fetch("/");
      if (!response.ok) throw new Error("Could not open Overview. Your camera is still connected.");
      const page = new DOMParser().parseFromString(await response.text(), "text/html");
      const appScript = page.querySelector('script[src*="/static/app.js"]');
      if (!page.querySelector("#overview") || !appScript) throw new Error("Overview could not be loaded. Please retry.");
      const scriptUrl = appScript.getAttribute("src");
      page.querySelectorAll("script").forEach(script => script.remove());
      overviewPage = document.createElement("div");
      overviewPage.id = "overview-page";
      overviewPage.className = "hidden";
      while (page.body.firstChild) overviewPage.append(page.body.firstChild);
      document.body.append(overviewPage);
      const script = document.createElement("script");
      script.src = scriptUrl;
      await new Promise((resolve, reject) => {
        script.onload = resolve;
        script.onerror = () => reject(new Error("Could not load the control room. Your camera is still connected."));
        overviewPage.append(script);
      });
    }
    cameraShell.classList.add("hidden");
    overviewPage.classList.remove("hidden");
    if (cam) {
      $("#live-session-bar").prepend($("#cam-video"));
      $("#cam-video").play().catch(() => {});
    }
    if (push) history.pushState({cameraView: "overview"}, "", "/#overview");
    document.title = "SnatchGuard | Control room";
    window.scrollTo(0, 0);
    render();
  } catch (error) {
    if (overviewPage && overviewPage.classList.contains("hidden")) { overviewPage.remove(); overviewPage = null; }
    message(error.message);
  } finally { overviewLoading = false; }
}

const sessionBar = document.createElement("aside");
sessionBar.id = "live-session-bar";
sessionBar.className = "live-session-bar hidden";
sessionBar.setAttribute("aria-label", "This device's camera");
sessionBar.innerHTML = '<div><strong id="overview-camera-status" role="status"></strong><button id="return-to-camera" class="action secondary">Back to camera</button><button id="overview-stop-camera" class="action secondary">Stop camera</button></div>';
document.body.append(sessionBar);
$("#return-to-camera").addEventListener("click", () => showFootage());
$("#overview-stop-camera").addEventListener("click", stop);
$("#open-control-room").addEventListener("click", event => {
  if (!cam) return; // Ordinary same-tab navigation when there is no active browser camera.
  event.preventDefault();
  showOverview();
});
document.addEventListener("click", event => {
  if (!overviewPage || overviewPage.classList.contains("hidden")) return;
  const link = event.target.closest("a");
  if (link?.getAttribute("href") === "/footage") { event.preventDefault(); showFootage(); }
});
window.addEventListener("popstate", () => {
  if (!overviewPage) return;
  if (location.pathname === "/footage") showFootage(false);
  else if (location.pathname === "/") showOverview(false);
});

status(); output();
setInterval(status, 2000);
setInterval(output, 2000);
window.addEventListener("pagehide", () => {
  if (cameraSession) fetch("/api/v1/browsercam/stop", {method: "POST", headers: cameraHeaders(), keepalive: true}).catch(() => {});
  stopCamera();
});

// SnatchGuard dashboard: live alert feed + camera health over Server-Sent Events.
"use strict";

const state = {
  events: new Map(),       // id -> event
  cameras: [],
  nodes: [],
  filter: "open",
  sound: true,
  openId: null,
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const LEVEL_LABEL = {
  CONFIRMED: "Chain snatching",
  SUSPECT: "Suspicious contact",
  TAMPER: "Camera tampering",
  TAMPER_CLEARED: "Tamper cleared",
  OFFLINE: "Edge node offline",
  ONLINE: "Edge node back online",
};
const TAMPER_TEXT = {
  blackout: "Image went black (lens covered or camera/IR failure)",
  blur: "Image lost focus (lens sprayed, smeared or defocused)",
  moved: "Camera view changed (camera turned or moved)",
  frozen: "Video frozen (stream stalled)",
};
const LIGHT_ICON = { day: "☀ Day", lowlight: "🌙 Low light", ir: "🌙 IR night" };
const STATUS_TEXT = { new: "Needs review", acknowledged: "Acknowledged", confirmed: "Confirmed", false_alarm: "False alarm" };

// ---------------------------------------------------------------------------
function timeAgo(iso) {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(iso).toLocaleString();
}
const clock = (iso) => (iso ? new Date(iso).toLocaleTimeString() : "–");

function isSystem(ev) { return ev.type !== "snatch"; }

function passesFilter(ev) {
  switch (state.filter) {
    case "open": return ev.status === "new";
    case "snatch": return ev.type === "snatch";
    case "system": return isSystem(ev);
    case "reviewed": return ev.status !== "new";
    default: return true;
  }
}

function summary(ev) {
  const p = ev.payload || {};
  if (ev.type === "tamper") return TAMPER_TEXT[p.condition] || p.condition;
  if (ev.type !== "snatch") return p.detail || "";
  const on = (p.evidence || []).filter((e) => e.passed && e.group !== "benign").map((e) => e.detail);
  return on.slice(0, 3).join(" · ");
}

// ---------------------------------------------------------------------------
function renderFeed() {
  const list = [...state.events.values()]
    .filter(passesFilter)
    .sort((a, b) => (b.received_at || "").localeCompare(a.received_at || ""));
  const feed = $("#feed");
  feed.innerHTML = list.map(cardHtml).join("");
  $("#feed-empty").classList.toggle("hidden", list.length > 0);
  feed.querySelectorAll(".card").forEach((el) => el.addEventListener("click", () => openDetail(el.dataset.id)));
}

function cardHtml(ev) {
  const p = ev.payload || {};
  const thumb = ev.snapshot
    ? `<img class="thumb" loading="lazy" src="${esc(ev.snapshot)}" alt="">`
    : `<div class="thumb placeholder">${ev.type === "snatch" ? "🎥" : "📡"}</div>`;
  const score = ev.type === "snatch" ? `<div class="score">${Math.round((ev.score || 0) * 100)}</div>` : "";
  const light = ev.lighting ? `<span class="pill">${esc(LIGHT_ICON[ev.lighting] || ev.lighting)}</span>` : "";
  const conf = ev.type === "snatch" && ev.confidence ? `<span class="pill">${esc(ev.confidence)} visibility</span>` : "";
  return `<article class="card ${esc(ev.level)} ${ev._fresh ? "fresh" : ""}" data-id="${esc(ev.id)}">
    <div class="bar"></div>
    ${thumb}
    <div class="body">
      <div class="title"><strong>${esc(LEVEL_LABEL[ev.level] || ev.level)}</strong>
        <span class="pill ${esc(ev.level)}">${esc(ev.level)}</span>${light}${conf}</div>
      <div class="meta">${esc(ev.camera_name || ev.camera_id || "")} · ${esc(ev.site || "")} · ${clock(ev.occurred_at)} (${timeAgo(ev.received_at)})</div>
      <div class="why">${esc(summary(ev))}</div>
    </div>
    <div class="side">${score}<span class="pill status-${esc(ev.status)}">${esc(STATUS_TEXT[ev.status] || ev.status)}</span></div>
  </article>`;
}

function renderCameras() {
  const box = $("#cams");
  const cams = state.cameras;
  $("#cams-empty").classList.toggle("hidden", cams.length > 0);
  const live = cams.filter((c) => c.status === "online").length;
  $("#k-cams").textContent = `${live}/${cams.length}`;
  $("#cam-summary").textContent = cams.length ? `${live} of ${cams.length} live` : "";
  box.innerHTML = cams.map((c) => {
    const p = c.payload || {};
    const online = c.status === "online";
    const badges = [];
    badges.push(`<span class="badge ${online ? "live" : "bad"}">${online ? "● LIVE" : esc((c.status || "unknown").toUpperCase())}</span>`);
    if (p.lighting) badges.push(`<span class="badge">${esc(LIGHT_ICON[p.lighting] || p.lighting)}</span>`);
    (p.tamper || []).forEach((t) => badges.push(`<span class="badge bad">⚠ ${esc(t)}</span>`));
    if (p.degraded) badges.push(`<span class="badge warn">⚠ low fps</span>`);
    const img = `/api/v1/cameras/${encodeURIComponent(c.node_id)}/${encodeURIComponent(c.camera_id)}/thumbnail.jpg?t=${Date.now()}`;
    return `<div class="cam">
      <div class="frame"><img src="${img}" alt="" onerror="this.style.visibility='hidden'">
        <div class="overlay">${badges.join("")}</div>
        ${online ? "" : `<div class="offline">NOT MONITORED</div>`}
      </div>
      <div class="info">
        <div style="min-width:0"><div class="name">${esc(c.name || c.camera_id)}</div>
          <div class="sub">${esc(c.node_id)} · ${esc(p.fps_proc ?? "–")} fps · ${esc(p.tracks ?? 0)} tracks · ${esc(p.pairs ?? 0)} pairs</div></div>
        <div class="sub" title="restarts">${p.restarts ? `↻ ${esc(p.restarts)}` : ""}</div>
      </div>
    </div>`;
  }).join("");
}

async function refreshStats() {
  try {
    const s = await (await fetch("/api/v1/stats")).json();
    $("#k-confirmed").textContent = s.confirmed;
    $("#k-suspect").textContent = s.suspect;
    $("#k-open").textContent = s.open;
  } catch (_) { /* offline: keep last values */ }
}

// ---------------------------------------------------------------------------
function openDetail(id) {
  const ev = state.events.get(id);
  if (!ev) return;
  state.openId = id;
  $("#modal-body").innerHTML = detailHtml(ev);
  $("#modal").classList.remove("hidden");
  wireDetail(ev);
}

function closeDetail() {
  state.openId = null;
  $("#modal").classList.add("hidden");
  $("#modal-body").innerHTML = "";
}

const GROUPS = [
  ["contact", "Contact"],
  ["escape", "Escape"],
  ["reaction", "Victim reaction"],
  ["approach", "Approach"],
];

function detailHtml(ev) {
  const p = ev.payload || {};
  const evidence = p.evidence || [];
  const checks = (group) => evidence.filter((e) => e.group === group)
    .map((e) => `<li class="${e.passed ? "on" : ""}"><span class="ic">${e.passed ? "✓" : "·"}</span><span>${esc(e.detail)}</span></li>`).join("");

  let right = "";
  if (ev.type === "snatch") {
    const comps = p.components || {};
    const pens = p.penalties || {};
    const bars = Object.entries(comps).map(([k, v]) => `<div class="brow"><span>${esc(k)}</span>
        <div class="track"><div class="fill" style="width:${Math.min(100, (v / 0.35) * 100)}%"></div></div><span class="val">+${v.toFixed(2)}</span></div>`).join("")
      + Object.entries(pens).map(([k, v]) => `<div class="brow neg"><span>${esc(k)}</span>
        <div class="track"><div class="fill" style="width:${Math.min(100, (v / 0.4) * 100)}%"></div></div><span class="val">−${Number(v).toFixed(2)}</span></div>`).join("");
    right = `
      <div class="section-title">Score ${Math.round((ev.score || 0) * 100)} / 100</div>
      <div class="breakdown">${bars}</div>
      ${GROUPS.map(([g, label]) => `<div class="section-title">${label}</div><ul class="checks">${checks(g)}</ul>`).join("")}
      <div class="section-title">Playfulness / familiarity checks (reduce the score)</div>
      <ul class="checks benign">${checks("benign")}</ul>`;
  } else if (ev.type === "tamper") {
    right = `<div class="section-title">Condition</div><p>${esc(TAMPER_TEXT[p.condition] || p.condition)}</p>
      <dl class="facts"><dt>Brightness</dt><dd>${esc(p.brightness)}</dd><dt>Sharpness</dt><dd>${esc(p.sharpness)}</dd>
      <dt>Scene match</dt><dd>${esc(p.scene_match)}</dd></dl>`;
  } else {
    right = `<p>${esc(p.detail || "")}</p>`;
  }

  const media = ev.snapshot || ev.clip
    ? `<div class="media-tabs">${ev.clip ? `<button data-media="clip" class="active">Clip</button>` : ""}${ev.snapshot ? `<button data-media="snap" class="${ev.clip ? "" : "active"}">Snapshot</button>` : ""}</div>
       <div id="media-clip" class="${ev.clip ? "" : "hidden"}">${ev.clip ? `<video src="${esc(ev.clip)}" controls autoplay muted loop playsinline></video>` : ""}</div>
       <div id="media-snap" class="${ev.clip ? "hidden" : ""}">${ev.snapshot ? `<img src="${esc(ev.snapshot)}" alt="snapshot">` : ""}</div>
       ${ev.type === "snatch" && !ev.clip ? `<div class="clip-wait">Evidence clip is uploading from the edge node…</div>` : ""}`
    : `<div class="clip-wait">No image for this event.</div>`;

  const lat = p.latency_s != null ? `${Number(p.latency_s).toFixed(1)} s after contact` : "–";
  return `
    <div class="detail-head">
      <h3>${esc(LEVEL_LABEL[ev.level] || ev.level)}</h3>
      <span class="pill ${esc(ev.level)}">${esc(ev.level)}</span>
      ${ev.lighting ? `<span class="pill">${esc(LIGHT_ICON[ev.lighting] || ev.lighting)}</span>` : ""}
      <span class="pill status-${esc(ev.status)}">${esc(STATUS_TEXT[ev.status] || ev.status)}</span>
    </div>
    <div class="detail">
      <div class="media">${media}
        <dl class="facts">
          <dt>Camera</dt><dd>${esc(ev.camera_name || ev.camera_id || "–")} (${esc(ev.node_id || "")})</dd>
          <dt>Site</dt><dd>${esc(ev.site || "–")}</dd>
          <dt>Happened</dt><dd>${esc(ev.occurred_at ? new Date(ev.occurred_at).toLocaleString() : "–")}</dd>
          <dt>Alert raised</dt><dd>${esc(lat)}</dd>
          ${ev.type === "snatch" ? `<dt>Visibility</dt><dd>${esc(p.confidence)} (${p.confidence === "high" ? "hand-to-neck seen by pose model" : "boxes/motion only"})</dd>
          <dt>Tracks</dt><dd>victim #${esc(p.victim_id)}, vehicle #${esc(p.vehicle_id)}, riders ${esc((p.rider_ids || []).map((r) => "#" + r).join(", ") || "–")}</dd>` : ""}
          <dt>Event id</dt><dd style="font-family:var(--mono);font-size:12px">${esc(ev.id)}</dd>
          ${ev.review_note ? `<dt>Review note</dt><dd>${esc(ev.review_note)}</dd>` : ""}
        </dl>
      </div>
      <div>
        ${right}
        <div class="review">
          <div class="section-title">Operator review</div>
          <textarea id="review-note" placeholder="Note (optional): what you saw, who was dispatched…">${esc(ev.review_note || "")}</textarea>
          <div class="btns">
            ${ev.type === "snatch" ? `<button class="btn red" data-review="confirmed">Confirm snatch</button>
            <button class="btn" data-review="false_alarm">False alarm</button>` : ""}
            <button class="btn green" data-review="acknowledged">Acknowledge</button>
          </div>
        </div>
      </div>
    </div>`;
}

function wireDetail(ev) {
  document.querySelectorAll("[data-media]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-media]").forEach((x) => x.classList.toggle("active", x === b));
    $("#media-clip")?.classList.toggle("hidden", b.dataset.media !== "clip");
    $("#media-snap")?.classList.toggle("hidden", b.dataset.media !== "snap");
  }));
  document.querySelectorAll("[data-review]").forEach((b) => b.addEventListener("click", async () => {
    const note = $("#review-note").value.trim() || null;
    const res = await fetch(`/api/v1/events/${encodeURIComponent(ev.id)}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: b.dataset.review, note }),
    });
    if (res.ok) { upsert(await res.json(), false); closeDetail(); refreshStats(); }
  }));
}

// ---------------------------------------------------------------------------
let audioCtx = null;
function beep(level) {
  if (!state.sound) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const pattern = level === "CONFIRMED" ? [880, 660, 880, 660] : [660];
    pattern.forEach((f, i) => {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.frequency.value = f;
      o.type = "square";
      g.gain.setValueAtTime(0.08, audioCtx.currentTime + i * 0.22);
      g.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + i * 0.22 + 0.2);
      o.connect(g).connect(audioCtx.destination);
      o.start(audioCtx.currentTime + i * 0.22);
      o.stop(audioCtx.currentTime + i * 0.22 + 0.21);
    });
  } catch (_) { /* audio blocked until the user interacts with the page */ }
}

function notify(ev) {
  const label = LEVEL_LABEL[ev.level] || ev.level;
  const where = ev.camera_name || ev.camera_id || ev.node_id || "";
  const t = document.createElement("div");
  t.className = `toast ${ev.level}`;
  t.innerHTML = `<strong>${esc(label)}</strong><span>${esc(where)} · ${clock(ev.occurred_at)}${ev.type === "snatch" ? ` · score ${Math.round((ev.score || 0) * 100)}` : ""}</span>`;
  t.addEventListener("click", () => { openDetail(ev.id); t.remove(); });
  $("#toasts").prepend(t);
  setTimeout(() => t.remove(), ev.level === "CONFIRMED" ? 30000 : 12000);
  if (["CONFIRMED", "SUSPECT", "TAMPER", "OFFLINE"].includes(ev.level)) beep(ev.level);
  if ("Notification" in window && Notification.permission === "granted" && ["CONFIRMED", "TAMPER", "OFFLINE"].includes(ev.level)) {
    new Notification(`SnatchGuard: ${label}`, { body: where, tag: ev.id });
  }
  if (ev.level === "CONFIRMED") flashTitle(label);
}

let flashTimer = null;
function flashTitle(text) {
  const base = "SnatchGuard Control";
  let n = 0;
  clearInterval(flashTimer);
  flashTimer = setInterval(() => {
    document.title = n % 2 ? base : `🚨 ${text}`;
    if (++n > 20 || document.hasFocus()) { clearInterval(flashTimer); document.title = base; }
  }, 800);
}

function upsert(ev, fresh) {
  const prev = state.events.get(ev.id);
  ev._fresh = fresh && !prev;
  state.events.set(ev.id, ev);
  renderFeed();
  if (state.openId === ev.id) { $("#modal-body").innerHTML = detailHtml(ev); wireDetail(ev); }
  if (ev._fresh) { notify(ev); setTimeout(() => { ev._fresh = false; }, 4000); }
}

// ---------------------------------------------------------------------------
function connect() {
  const es = new EventSource("/api/v1/stream");
  const conn = $("#conn");
  es.onopen = () => { conn.className = "conn live"; $("#conn-text").textContent = "Live"; loadInitial(); };
  es.onerror = () => { conn.className = "conn down"; $("#conn-text").textContent = "Reconnecting…"; };
  es.addEventListener("event", (m) => { upsert(JSON.parse(m.data), true); refreshStats(); });
  es.addEventListener("event_update", (m) => upsert(JSON.parse(m.data), false));
  es.addEventListener("cameras", (m) => {
    const d = JSON.parse(m.data);
    state.cameras = d.cameras || [];
    renderCameras();
  });
}

async function loadInitial() {
  const [events, cams] = await Promise.all([
    fetch("/api/v1/events?limit=200").then((r) => r.json()),
    fetch("/api/v1/cameras").then((r) => r.json()),
  ]);
  events.forEach((ev) => { if (!state.events.has(ev.id)) state.events.set(ev.id, ev); });
  state.cameras = cams.cameras || [];
  renderFeed();
  renderCameras();
  refreshStats();
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  state.filter = b.dataset.filter;
  renderFeed();
}));
$("#modal-close").addEventListener("click", closeDetail);
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeDetail(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
$("#sound-btn").addEventListener("click", () => {
  state.sound = !state.sound;
  $("#sound-btn").textContent = state.sound ? "🔔 Sound on" : "🔕 Sound off";
  if (state.sound) beep("SUSPECT");
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
});
document.addEventListener("click", () => {
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
}, { once: true });
setInterval(renderFeed, 30000);   // keep "x min ago" fresh
loadInitial().catch(() => {});    // show stored alerts even if the live stream is blocked
connect();

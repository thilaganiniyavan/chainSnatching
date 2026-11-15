// SnatchGuard dashboard: live alert feed + camera health over Server-Sent Events.
"use strict";

const state = {
  events: new Map(),       // id -> event
  cameras: [],
  nodes: [],
  filter: "all",
  sound: true,
  openId: null,
  risk: new Map(),         // node/camera -> {risk, learned, at} from the edge's learned scorer
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const LEVEL_LABEL = {
  CONFIRMED: "High-priority incident",
  SUSPECT: "Suspected incident",
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
const MODE_TEXT = { motorcycle: "by motorcycle", bicycle: "by bicycle", on_foot: "on foot" };
const displaySite = (ev) => ev.payload?.recorded_example ? "Recorded CCTV" : (ev.site || "");
const modeOf = (ev) => (ev.payload && MODE_TEXT[ev.payload.mode]) || "";
const titleOf = (ev) => (ev.payload?.recorded_example ? "CCTV incident example" : (LEVEL_LABEL[ev.level] || ev.level)) + (ev.type === "snatch" && modeOf(ev) ? ` (${modeOf(ev)})` : "");
const STATUS_TEXT = { new: "Needs review", acknowledged: "Acknowledged", confirmed: "Confirmed", false_alarm: "False alarm" };
const SOURCE_TEXT = { rules: "rules", model: "learned scorer", fused: "rules + model", verifier: "learned verifier" };
const pct100 = (x) => Math.round(Number(x) * 100);
function sourcePills(ev) {
  const p = ev.payload || {};
  if (ev.type !== "snatch") return "";
  let out = p.recorded_example ? `<span class="pill">RECORDED CCTV</span>` : "";
  if (p.source && p.source !== "rules") out += `<span class="pill src-${esc(p.source)}">${esc(SOURCE_TEXT[p.source] || p.source)}</span>`;
  if (p.model_score != null) out += `<span class="pill" title="learned scorer">model ${pct100(p.model_score)}</span>`;
  if (p.supersedes) out += `<span class="pill" title="upgrades an earlier alert">upgrade</span>`;
  return out;
}

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

function isSynthetic(ev) { return ev.node_id === "edge-sim-01" || ev.site === "Scripted demonstration"; }
function passesFilter(ev) {
  if (isSynthetic(ev)) return false;
  const query = $("#search").value.trim().toLowerCase();
  if (query && ![ev.camera_name, ev.camera_id, ev.site, ev.level, ev.type, ev.status, titleOf(ev)].some((v) => String(v || "").toLowerCase().includes(query))) return false;
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
  if (p.recorded_example) return "Review the labelled incident window and actor tracks.";
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
      <div class="title"><strong>${esc(titleOf(ev))}</strong>
        <span class="pill ${esc(ev.level)}">${esc(ev.level === "CONFIRMED" ? "HIGH PRIORITY" : ev.level)}</span>${light}${conf}${sourcePills(ev)}</div>
      <div class="meta">${esc(ev.camera_name || ev.camera_id || "")} · ${esc(displaySite(ev))} · ${p.recorded_example ? "Recorded example · event time unknown" : `${clock(ev.occurred_at)} (${timeAgo(ev.received_at)})`}</div>
      <div class="why">${esc(summary(ev))}</div>
    </div>
    <div class="side">${score}<span class="pill status-${esc(ev.status)}">${esc(p.recorded_example ? "Recorded example" : (STATUS_TEXT[ev.status] || ev.status))}</span></div>
  </article>`;
}

function renderCameras() {
  const box = $("#cams");
  const cams = state.cameras.filter((c) => !isSynthetic(c));
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
    if (["error", "missing"].includes(p.verifier)) badges.push(`<span class="badge bad">⚠ verifier ${esc(p.verifier)}</span>`);
    if (p.verifier_stats?.dropped) badges.push(`<span class="badge warn">${esc(p.verifier_stats.dropped)} clips dropped</span>`);
    const img = `/api/v1/cameras/${encodeURIComponent(c.node_id)}/${encodeURIComponent(c.camera_id)}/thumbnail.jpg?t=${Date.now()}`;
    const key = `${c.node_id}/${c.camera_id}`;
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
      <div class="risk" data-risk-key="${esc(key)}">${riskInner(riskOf(key, p))}</div>
    </div>`;
  }).join("");
}

function riskOf(key, payload) {
  const live = state.risk.get(key);
  if (live) return live;
  return { risk: payload ? payload.risk : null, learned: payload ? payload.learned : null };
}

function riskInner(r) {
  if (!r || r.learned == null || r.learned === "off") return "";
  if (r.learned !== "on") return `<span class="risk-label muted">learned scorer: ${esc(r.learned)} (rules only)</span>`;
  const v = r.risk == null ? 0 : Math.max(0, Math.min(1, Number(r.risk)));
  const cls = v >= 0.7 ? "high" : v >= 0.4 ? "mid" : "low";
  return `<div class="risk-track"><div class="risk-fill ${cls}" style="width:${(v * 100).toFixed(0)}%"></div></div>
    <span class="risk-label">model score ${r.risk == null ? "–" : pct100(v)}</span>`;
}

function updateRisk(entries) {
  (entries || []).forEach((e) => state.risk.set(`${e.node_id}/${e.camera_id}`, e));
  document.querySelectorAll("[data-risk-key]").forEach((el) => {
    const r = state.risk.get(el.dataset.riskKey);
    if (r) el.innerHTML = riskInner(r);
  });
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
    .map((e) => `<li class="${e.passed ? "on" : ""}"><span class="ic">${e.passed ? "✓" : "·"}</span><span>${esc(p.recorded_example && e.group === "model" ? "Verifier scored a candidate from the recorded clip." : e.detail)}</span></li>`).join("");

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
      <ul class="checks benign">${checks("benign")}</ul>
      ${p.source || p.model_score != null ? `<div class="section-title">Learned scorer</div>
      <dl class="facts"><dt>Decided by</dt><dd>${esc(SOURCE_TEXT[p.source] || p.source || "rules")}</dd>
        <dt>Model score</dt><dd>${p.model_score == null ? "– (not available)" : pct100(p.model_score) + " / 100"}</dd>
        <dt>Rule verdict</dt><dd>${p.rule_level ? esc(p.rule_level) + (p.rule_score != null ? ` (score ${pct100(p.rule_score)})` : "") : "–"}</dd>
        ${p.supersedes ? `<dt>Upgrades</dt><dd style="font-family:var(--mono);font-size:12px">${esc(p.supersedes)}</dd>` : ""}
      </dl><ul class="checks">${checks("model")}</ul>` : ""}`;
  } else if (ev.type === "tamper") {
    right = `<div class="section-title">Condition</div><p>${esc(TAMPER_TEXT[p.condition] || p.condition)}</p>
      <dl class="facts"><dt>Brightness</dt><dd>${esc(p.brightness)}</dd><dt>Sharpness</dt><dd>${esc(p.sharpness)}</dd>
      <dt>Scene match</dt><dd>${esc(p.scene_match)}</dd></dl>`;
  } else {
    right = `<p>${esc(p.detail || "")}</p>`;
  }

  const ann = p.incident_annotation;
  const media = ev.snapshot || ev.clip
    ? `<div class="media-tabs">${ev.clip ? `<button data-media="clip" class="active">Clip</button>` : ""}${ev.snapshot ? `<button data-media="snap" class="${ev.clip ? "" : "active"}">Snapshot</button>` : ""}</div>
       <div id="media-clip" class="${ev.clip ? "" : "hidden"}">${ev.clip ? `<div class="video-stage"><video id="evidence-video" src="${esc(ev.clip)}" controls muted playsinline></video><div id="incident-region" class="incident-region hidden"><span>ACTOR TRACKS</span></div></div>${ann ? `<div class="incident-timeline"><div class="timeline-heading"><span><i></i> LABELLED INCIDENT WINDOW</span><strong>${Number(ann.start_s).toFixed(1)}s – ${Number(ann.end_s).toFixed(1)}s</strong></div><div class="timeline-track"><div id="incident-window" class="timeline-window"></div><div id="timeline-playhead" class="timeline-playhead"></div></div><div class="timeline-caption"><span>Clip start</span><button id="jump-incident" class="jump-incident">↗ Jump to incident</button><span id="clip-duration">–</span></div><p>Highlighted region follows cached actor tracks where both actors are visible. Event timing is labelled footage, not a verified live detection time.</p></div>` : `<p class="annotation-unavailable">No labelled event timestamp is available for this clip. Live evidence may already include tracking overlays.</p>`}` : ""}</div>
       <div id="media-snap" class="${ev.clip ? "hidden" : ""}">${ev.snapshot ? `<img src="${esc(ev.snapshot)}" alt="snapshot">` : ""}</div>
       ${ev.type === "snatch" && !ev.clip ? `<div class="clip-wait">Evidence clip is not available yet.</div>` : ""}`
    : `<div class="clip-wait">No image for this event.</div>`;

  const lat = p.latency_s != null ? `${Number(p.latency_s).toFixed(1)} s after contact` : "–";
  return `
    ${p.recorded_example ? `<div class="notice">Previously analysed CCTV footage. The highlighted interval is a labelled event window; this entry is not a live alert.</div>` : ""}
    <div class="detail-head">
      <h3>${esc(titleOf(ev))}</h3>
      <span class="pill ${esc(ev.level)}">${esc(ev.level === "CONFIRMED" ? "HIGH PRIORITY" : ev.level)}</span>
      ${ev.lighting ? `<span class="pill">${esc(LIGHT_ICON[ev.lighting] || ev.lighting)}</span>` : ""}
      <span class="pill status-${esc(ev.status)}">${esc(STATUS_TEXT[ev.status] || ev.status)}</span>
    </div>
    <div class="detail">
      <div class="media">${media}
        <dl class="facts">
          <dt>Camera</dt><dd>${esc(ev.camera_name || ev.camera_id || "–")} (${esc(ev.node_id || "")})</dd>
          <dt>Site</dt><dd>${esc(displaySite(ev) || "–")}</dd>
          <dt>Happened</dt><dd>${esc(ev.occurred_at ? new Date(ev.occurred_at).toLocaleString() : "–")}</dd>
          <dt>Alert raised</dt><dd>${esc(lat)}</dd>
          ${ev.type === "snatch" ? `<dt>Visibility</dt><dd>${esc(p.confidence)} (${p.confidence === "high" ? "hand-to-neck seen by pose model" : "boxes/motion only"})</dd>
          <dt>Tracks</dt><dd>victim #${esc(p.victim_id)}, ${p.mode === "on_foot" ? "attacker" : "vehicle"} #${esc(p.vehicle_id)}${p.mode === "on_foot" ? "" : `, riders ${esc((p.rider_ids || []).map((r) => "#" + r).join(", ") || "–")}`}</dd>` : ""}
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
  const video = $("#evidence-video");
  const annotation = ev.payload?.incident_annotation;
  if (video && annotation) {
    const update = () => {
      const duration = video.duration;
      if (Number.isFinite(duration) && duration > 0) {
        const left = Math.max(0, Math.min(100, 100 * annotation.start_s / duration));
        const right = Math.max(left, Math.min(100, 100 * annotation.end_s / duration));
        $("#incident-window").style.left = `${left}%`;
        $("#incident-window").style.width = `${right - left}%`;
        $("#timeline-playhead").style.left = `${Math.min(100, 100 * video.currentTime / duration)}%`;
        $("#clip-duration").textContent = `${duration.toFixed(1)}s`;
      }
      const regions = annotation.regions || [];
      let closest = null;
      for (const region of regions) if (!closest || Math.abs(region.t - video.currentTime) < Math.abs(closest.t - video.currentTime)) closest = region;
      const overlay = $("#incident-region");
      const show = closest && Math.abs(closest.t - video.currentTime) <= 0.3 && video.currentTime >= annotation.start_s && video.currentTime <= annotation.end_s;
      overlay.classList.toggle("hidden", !show);
      if (show) {
        const [x1, y1, x2, y2] = closest.box;
        Object.assign(overlay.style, {left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`});
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    $("#jump-incident").addEventListener("click", () => { video.currentTime = Math.max(0, annotation.start_s); video.play().catch(() => {}); });
    update();
  }
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
    else { alert("Review could not be saved. Please try again."); }
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
  const label = titleOf(ev);
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
  ev._fresh = fresh && !prev && !isSynthetic(ev);
  state.events.set(ev.id, ev);
  renderFeed();
  if (state.openId === ev.id) { $("#modal-body").innerHTML = detailHtml(ev); wireDetail(ev); }
  if (ev._fresh) { notify(ev); setTimeout(() => { ev._fresh = false; }, 4000); }
}

// ---------------------------------------------------------------------------
function connect() {
  const es = new EventSource("/api/v1/stream");
  const conn = $("#conn");
  es.onopen = () => { document.querySelector(".live-tag").textContent = "● LIVE FEED"; conn.className = "conn live"; $("#conn-text").textContent = "Live"; loadInitial(); };
  es.onerror = () => { document.querySelector(".live-tag").textContent = "FEED DISCONNECTED"; conn.className = "conn down"; $("#conn-text").textContent = "Reconnecting…"; };
  es.addEventListener("event", (m) => { upsert(JSON.parse(m.data), true); refreshStats(); });
  es.addEventListener("event_update", (m) => upsert(JSON.parse(m.data), false));
  es.addEventListener("risk", (m) => updateRisk(JSON.parse(m.data).cameras));
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
  fetch("/api/v1/live").then((r) => r.json()).then((d) => updateRisk(d.cameras)).catch(() => {});
}

$("#search").addEventListener("input", renderFeed);
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

$("#start-real-demo").addEventListener("click", async () => {
  const button = $("#start-real-demo");
  const message = $("#demo-message");
  button.disabled = true;
  message.textContent = "Starting recorded footage…";
  try {
    const response = await fetch("/api/v1/footage/real-demo", {method: "POST"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Unable to start demonstration");
    message.textContent = "Processing real footage. Camera updates and alerts will appear below.";
  } catch (error) { message.textContent = error.message || "Could not start demonstration."; }
  finally { button.disabled = false; }
});

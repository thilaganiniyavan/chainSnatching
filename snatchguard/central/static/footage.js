"use strict";
const $ = (s) => document.querySelector(s);
const message = (s) => { $("#message").textContent = s; $("#message").hidden = !s; };
async function status() {
  try {
    const res = await fetch("/api/v1/footage");
    if (!res.ok) throw new Error("Status unavailable");
    const data = await res.json();
    $("#status-title").textContent = ({ idle: "Ready to start", running: "Processing in progress", completed: "Processing completed", stopped: "Processing stopped", failed: "Processing failed" })[data.state] || data.state;
    $("#status-detail").textContent = data.name ? `${data.name} · ${data.kind}` : "Only one video can run at a time.";
    $("#stop").hidden = data.state !== "running";
    $("#real-demo").disabled = data.state === "running";
    $("#start-live").disabled = data.state === "running" || !$("#model").value;
    $("#job-log").hidden = !data.message;
    $("#job-log").textContent = data.message || "";
  } catch (_) { $("#status-title").textContent = "Server unavailable"; }
}
async function send(url, options) {
  message("");
  try {
    const res = await fetch(url, options);
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Request failed");
    message("Started. Open the dashboard to see camera updates and alerts.");
    await status();
    return true;
  } catch (err) { message(err.message || "The request failed. Please try again."); return false; }
}
$("#real-demo").addEventListener("click", () => send("/api/v1/footage/real-demo", {method:"POST"}));
$("#stop").addEventListener("click", async () => { if (await send("/api/v1/footage/stop", {method:"POST"})) message("Stop requested. The processing job is shutting down."); });
$("#video").addEventListener("change", (e) => { $("#filename").textContent = e.target.files[0]?.name || "No file selected"; });
$("#upload-form").addEventListener("submit", async (e) => { e.preventDefault(); const form = new FormData(); form.append("video", $("#video").files[0]); message("Uploading video…"); await send("/api/v1/footage/upload", {method:"POST", body:form}); });
status(); setInterval(status, 3000);

async function loadModels() {
  try {
    const response = await fetch("/api/v1/footage/models");
    if (!response.ok) throw new Error();
    const data = await response.json();
    const select = $("#model");
    select.replaceChildren();
    for (const name of data.models) { const option = document.createElement("option"); option.value = name; option.textContent = name; select.append(option); }
    $("#model-help").textContent = data.models.length ? "Only use checkpoints from a trusted source." : "No verifier checkpoints found in models/verifier.";
    await status();
  } catch (_) { $("#model-help").textContent = "Could not load models."; $("#start-live").disabled = true; }
}
$("#live-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send("/api/v1/footage/live", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({source: $("#camera-source").value, name: $("#camera-name").value, model: $("#model").value})});
});
loadModels();

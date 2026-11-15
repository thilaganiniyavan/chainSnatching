"use strict";
const $ = (s) => document.querySelector(s);
const message = (s) => { $("#message").textContent = s; $("#message").hidden = !s; };
async function status() {
  try {
    const res = await fetch("/api/v1/footage");
    if (!res.ok) throw new Error("Status unavailable");
    const data = await res.json();
    $("#status-title").textContent = ({ idle: "Ready to start", running: "Processing in progress", completed: "Processing completed", failed: "Processing failed" })[data.state] || data.state;
    $("#status-detail").textContent = data.name ? `${data.name} · ${data.kind}` : "Only one video can run at a time.";
    $("#stop").hidden = data.state !== "running";
    $("#real-demo").disabled = data.state === "running";
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
  } catch (err) { message(err.message || "The request failed. Please try again."); }
}
$("#real-demo").addEventListener("click", () => send("/api/v1/footage/real-demo", {method:"POST"}));
$("#stop").addEventListener("click", async () => { await send("/api/v1/footage/stop", {method:"POST"}); message("Stop requested. The processing job is shutting down."); });
$("#video").addEventListener("change", (e) => { $("#filename").textContent = e.target.files[0]?.name || "No file selected"; });
$("#upload-form").addEventListener("submit", async (e) => { e.preventDefault(); const form = new FormData(); form.append("video", $("#video").files[0]); message("Uploading video…"); await send("/api/v1/footage/upload", {method:"POST", body:form}); });
status(); setInterval(status, 3000);

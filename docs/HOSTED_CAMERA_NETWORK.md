# Hosted camera network

Each browser that selects **Start live CCTV** on Footage & demo gets its own camera stream and detection process. Enter a camera name and a site before starting. The shared control room lists these cameras and receives their alerts through the existing event stream, alongside edge cameras. Merely opening a page does not enable the camera: the user must allow camera access and start monitoring. Keep the page open and the device awake; browser suspension or network loss interrupts capture. While a camera is active, **Open control room** switches to Overview in the same tab without unloading the stream. A floating preview shows this device’s status, with Stop camera and Back to camera controls. Leaving the document, refreshing or closing the tab stops capture.

## Hosting setup

Use a persistent Python server or container with installed detection dependencies, the configured detector / pose / verifier weights, writable persistent storage, and sufficient CPU or GPU capacity. Static website hosting and request-only serverless functions cannot run these long-lived detector processes. Install the project's central, edge and learned dependency groups. Provision model weights separately; they are not bundled with the repository.

Run **one server process / one worker**. Camera sessions, WebSocket frames and event subscriptions currently live in process memory. Multiple workers or replicas require a shared stream broker and distributed job scheduling before use. Restarting the server ends browser-camera sessions; reconnect from each device. Saved events, clips and operator links remain on persistent disk.

Set these environment variables in the deployment's secret / environment settings:

| Variable | Purpose |
| --- | --- |
| `SNATCHGUARD_API_KEYS` | A private random edge ingestion key; replace the development default. |
| `SNATCHGUARD_DASHBOARD_PASSWORD` | Password required to view and control the portal and issue camera sessions. |
| `SNATCHGUARD_DASHBOARD_USER` | Operator username (default `operator`). |
| `SNATCHGUARD_MAX_BROWSER_CAMERAS` | Concurrent browser-session capacity; defaults to 4. Each active device runs a detector; increase only after a load test. |
| `SNATCHGUARD_INTERNAL_PORT` | Actual backend listening port, if the proxy's request scope reports a different port. |
| `SNATCHGUARD_INTERNAL_URL` | Optional direct backend origin, e.g. `http://127.0.0.1:8000`; used by local detectors for frames, alerts and heartbeats. Must reach this same process, avoiding a load balancer across replicas. |

Example backend command:

```sh
python -m snatchguard.central --host 0.0.0.0 --port 8000 --data /persistent/snatchguard
```

Terminate HTTPS at the hosting proxy. Forward HTTP, WebSocket upgrades for `/api/v1/browsercam/ws`, and the original Host header. Use WebSocket idle timeouts above 30 seconds and disable proxy buffering on `/api/v1/stream` (server-sent events). Browser frames use same-origin `wss://` on HTTPS. The detector's internal MJPEG stream is protected by the edge key. Never expose model config files or the private footage directory as public static files.

Each session has a random owner token. Its start, status, output and stop operations require that token and session ID; closing one tab cannot stop another device's detector. Camera buffers are isolated, retain up to 40 seconds and are capped at 32 MiB per connection. Tokens awaiting a camera expire after 60 seconds. A stream with no frames for 15 seconds is stopped and marked offline. Closed-session buffers are cleared after 60 seconds; session metadata expires after an hour. This is bounded evidence buffering rather than continuous video archival.

## Cross-camera investigation

Open a snatch alert in the control room and use **Across cameras**. Suggestions contain alerts from a different camera at the same named site within two minutes (latest 500 visible alerts searched). Generic `Live camera` and uploaded footage locations are excluded. Assign the same site name only to cameras in the same area. Compare snapshots and clips, choose **Related incident** or **Possible same person (operator assessment)**, add an evidence note and save the link. Links are bidirectional, stored in SQLite and removable.

These are time-and-site suggestions, not automatic person identification. The detector's track IDs are camera-local and must never be matched by number across cameras. Reliable automatic person re-identification would require a separate model, multi-camera evaluation, camera topology / travel-time constraints and a calibrated review workflow. It is not implemented by this change. Two cameras can capture an event without either camera generating an alert; those unflagged sightings do not appear in the suggestion list.

## Verification before sharing the hosted URL

1. Open the HTTPS site on two devices, log in, give them different camera names and the same site, and allow camera access.
2. Start both cameras. Confirm two separate network entries and thumbnails in the control room.
3. Stop or disconnect one device. Confirm that it becomes offline while the other continues monitoring.
4. Use appropriately labelled test footage to check event accuracy, alert delivery, clip playback and cross-camera links. Automated integration tests mock detectors; they do not establish model accuracy or concurrent processing throughput.

Browser camera requirements: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

## Alert controls and clean deployments

The trash icon moves the entire alert to the **Deleted** filter. Deleted alerts disappear from All, Open, Incidents, Camera health, Reviewed and Detected snatches. Restore is available only in Deleted. The incident and original evidence remain stored, and a late clip upload or completed trim cannot move the alert back into active views. Previously removed videos and dismissed tampering alerts are also treated as Deleted.

Select **Enable sound** once in the control room. New snatch detections sound immediately, including candidates awaiting review; event updates and duplicate messages do not repeat the sound. The Sound button can mute alerts. Browser autoplay restrictions apply until a user interacts with the page.

Camera disconnects update Camera status without adding edge offline / online alerts or sounds to the incident feed. Uploaded recordings and demonstration sources are excluded from the camera network; disconnected entries age out of the current network after five minutes.

For a clean deployment, use a new persistent data directory. To archive old activity in an existing directory without deleting media:

```sh
python -m snatchguard.central.cleanup --data runtime/central
```

Restore that archived activity if needed:

```sh
python -m snatchguard.central.cleanup --data runtime/central --restore
```

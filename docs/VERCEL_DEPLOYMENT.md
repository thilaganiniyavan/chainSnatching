# Deploying the portal to Vercel

The Vercel build serves the public dashboard, footage page and metrics tab. API, event-stream and evidence requests are forwarded to a separately hosted SnatchGuard backend. Browser camera frames connect directly to that backend over WSS using the per-session token issued through the authenticated API.

This configuration does not run the detector inside a Vercel Function. The current backend owns camera buffers, child detector processes, SQLite and evidence files. A Vercel-only deployment needs a separate migration to durable database / object storage, shared session transport and worker scheduling. Vercel's current WebSocket and daemon support does not automatically migrate that application state.

## 1. Prepare the detector backend

Follow [Hosted camera network](HOSTED_CAMERA_NETWORK.md). Run one process on a persistent server with the installed weights and storage. Give it a public HTTPS URL with WebSocket support. Keep the dashboard password enabled and replace the development ingestion key.

Set on that backend:

```text
SNATCHGUARD_CAMERA_ORIGINS=https://your-project.vercel.app
```

Use the exact production portal origin, with no trailing path. Multiple approved origins can be comma-separated. Unknown preview origins are rejected; add only specific preview URLs you intend to test. The ordinary single-origin local installation needs no new setting.

## 2. Configure Vercel

Import the source repository, or deploy this project directory using the Vercel connection / CLI. Use:

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Root directory | Repository root |
| Build command | `node scripts/build_vercel.mjs` (provided in `vercel.json`) |
| Output directory | Leave unset; the build emits the Vercel Build Output API layout |
| Install command | Empty (provided in `vercel.json`); the portal build has no package dependencies |
| Environment variable | `SNATCHGUARD_BACKEND_URL=https://your-detector-server.example` |

The backend value must be a public HTTPS origin, without a path, credentials or query. The build fails if it is missing rather than publishing a portal with broken camera APIs. Do not enter the edge API key or dashboard password in this variable. The browser accesses the API through the Vercel domain and uses the backend's existing dashboard login.

`.vercelignore` restricts uploaded source to the build script, configuration and six public portal files. Datasets, model weights, reports, runtime databases, private evidence and local configuration are excluded. The generated static build also uses an explicit allowlist. Nothing from `models/` or `runtime/` is deployed to the Vercel CDN.

## 3. Verify the deployment

1. Load `/`, `/footage`, and `/metrics`. Metrics also remain available at `/static/metrics.html` for existing links.
2. Check `/healthz` through the deployed portal. This should reach the detector backend.
3. Start cameras on two devices, approve access and check both network entries in a separate control-room tab.
4. Verify alerts, evidence playback, cross-camera links and disconnection behavior. The backend must permit the deployed portal's exact WebSocket Origin.
5. Test expected video-upload sizes and durations against the deployed proxy and backend. This split deployment has not been validated live until a real backend URL and Vercel project are supplied.

Changing `SNATCHGUARD_BACKEND_URL` requires a new Vercel build. Backend secrets and model weights stay on the backend. The static portal HTML is public; API responses and evidence remain protected by the backend dashboard authentication and are marked `private, no-store` in proxy routing.

## Local deployment checks

```sh
node tests/test_vercel_build.mjs
node tests/test_browser_camera_ui.js
python -m pytest tests/test_browser_network.py tests/test_footage_live.py tests/test_central.py
```

References:
- https://vercel.com/docs/build-output-api/configuration
- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/background-workers

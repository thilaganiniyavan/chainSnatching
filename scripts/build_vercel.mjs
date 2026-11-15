/** Build only the public portal; detection and private evidence remain on the backend. */
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

export function backendOrigin(value) {
  if (!value) throw new Error('Set SNATCHGUARD_BACKEND_URL to the public HTTPS detector server before deploying.');
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('SNATCHGUARD_BACKEND_URL must be an HTTPS origin without credentials, path or query.');
  }
  if (['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    throw new Error('The detector backend must be publicly reachable; localhost will not work from Vercel.');
  }
  return url.origin;
}

export async function build({root, output, backend}) {
  const origin = backendOrigin(backend);
  const source = join(root, 'snatchguard/central/static');
  const publicDir = join(output, 'static');
  await mkdir(join(publicDir, 'static'), {recursive: true});
  // Explicit allowlist: never upload datasets, model weights, configs, SQLite or clips.
  for (const name of ['index.html', 'footage.html', 'metrics.html', 'app.js', 'footage.js', 'style.css']) {
    let content = await readFile(join(source, name), 'utf8');
    // Camera bytes go straight to the detector, independent of proxy WebSocket support.
    if (name === 'footage.html') {
      const wsOrigin = origin.replace(/^https:/, 'wss:');
      content = content.replace('<head>', `<head><meta name="snatchguard-camera-origin" content="${wsOrigin}">`);
    }
    await writeFile(join(publicDir, 'static', name), content);
  }
  const routes = [
    {src: '/api/(.*)', dest: `${origin}/api/$1`, headers: {'Cache-Control': 'private, no-store'}},
    {src: '/media/(.*)', dest: `${origin}/media/$1`, headers: {'Cache-Control': 'private, no-store'}},
    {src: '/healthz', dest: `${origin}/healthz`, headers: {'Cache-Control': 'no-store'}},
    {src: '/', dest: '/static/index.html'},
    {src: '/footage/?', dest: '/static/footage.html'},
    {src: '/metrics/?', dest: '/static/metrics.html'},
    {handle: 'filesystem'},
    {src: '/.*', status: 404},
  ];
  await writeFile(join(output, 'config.json'), JSON.stringify({version: 3, routes}, null, 2) + '\n');
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  await build({root, output: join(root, '.vercel/output'), backend: process.env.SNATCHGUARD_BACKEND_URL});
  console.log('Built Vercel portal with API and evidence forwarding.');
}

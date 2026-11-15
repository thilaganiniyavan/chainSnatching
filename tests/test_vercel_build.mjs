import assert from 'node:assert/strict';
import {mkdtemp, readFile, readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, join} from 'node:path';
import {backendOrigin, build} from '../scripts/build_vercel.mjs';

for (const value of [undefined, 'http://detector.example', 'https://user:pass@detector.example', 'https://127.0.0.1', 'https://detector.example/path', 'https://detector.example?key=secret']) {
  assert.throws(() => backendOrigin(value));
}
const output = await mkdtemp(join(tmpdir(), 'snatchguard-vercel-test-'));
await build({root: resolve('.'), output, backend: 'https://detector.example'});
const config = JSON.parse(await readFile(join(output, 'config.json'), 'utf8'));
assert.equal(config.version, 3);
assert.equal(config.routes.find(r => r.src === '/api/(.*)').dest, 'https://detector.example/api/$1');
assert.equal(config.routes.find(r => r.src === '/media/(.*)').headers['Cache-Control'], 'private, no-store');
assert.equal(config.routes.find(r => r.src === '/metrics/?').dest, '/static/metrics.html');
assert.equal(config.routes.at(-1).status, 404);
assert.match(await readFile(join(output, 'static/static/footage.html'), 'utf8'), /content="wss:\/\/detector.example"/);
assert.deepEqual((await readdir(join(output, 'static/static'))).sort(), ['app.js', 'footage.html', 'footage.js', 'index.html', 'metrics.html', 'style.css']);
console.log('Vercel build passed: HTTPS origin validation, private API routes, page routes and public-file allowlist.');

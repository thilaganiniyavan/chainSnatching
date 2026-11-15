// Browser lifecycle contract without a camera, network or external packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const elements = new Map();
const calls = [];
const socketUrls = [];
const timers = [];
const navigations = [];
let tracksStopped = 0;
let sentFrames = 0;
const hidden = new Set();
const classList = {toggle(name, on) { if (on) hidden.add(name); else hidden.delete(name); }, add(name) {hidden.add(name);}, remove(name) {hidden.delete(name);}, contains(name) {return hidden.has(name);}};
const shell = {classList};
const view = {prepend() {}};
function element(selector) {
  if (selector === '.shell') return shell;
  if (selector === '#live-view') return view;
  if (selector.startsWith("meta[")) return process.env.TEST_CAMERA_ORIGIN ? {content: process.env.TEST_CAMERA_ORIGIN} : null;
  if (!elements.has(selector)) elements.set(selector, {
    value: selector === '#camera-name' ? 'Gate' : 'Campus', hidden: false, disabled: false,
    videoWidth: 960, videoHeight: 540, classList: {toggle() {}, add() {}, remove() {}, contains() {return false;}}, files: [],
    handlers: {}, addEventListener(name, fn) { this.handlers[name] = fn; },
    play: async () => {}, children: [], replaceChildren() {this.children = [];}, append(...children) {this.children.push(...children);},
  });
  return elements.get(selector);
}
const lifecycle = {};
class Socket {
  constructor(url) { socketUrls.push(url); this.url = url; this.readyState = 1; this.bufferedAmount = 0; queueMicrotask(() => this.onopen()); }
  send() { sentFrames++; queueMicrotask(() => this.onmessage({data: '{"ready":true}'})); }
  close() { if (this.readyState !== 1) return; this.readyState = 3; queueMicrotask(() => this.onclose?.()); }
}
const context = vm.createContext({
  console, setTimeout, clearTimeout, setInterval: (fn, ms) => { timers.push({fn, ms}); return timers.length; }, clearInterval() {},
  FormData: class {append() {}},
  history: {pushState(state, unused, url) { navigations.push(url); }},
  location: {pathname: "/footage", protocol: 'https:', host: 'portal.example'}, WebSocket: Socket,
  navigator: {mediaDevices: {getUserMedia: async () => ({getTracks: () => [{stop() {tracksStopped++;}}]})}},
  window: {scrollTo() {}, addEventListener: (name, fn) => { lifecycle[name] = fn; }},
  document: {
    querySelector: element, addEventListener() {}, body: {append(node) { if (node.id) elements.set('#' + node.id, node); }},
    createElement: tag => {
      const node = {tag, children: [], className: '', setAttribute() {}, prepend() {}, handlers: {}, addEventListener(name, fn) {this.handlers[name] = fn;},
        append(...children) {this.children.push(...children); for (const child of children) if (child?.tag === 'script') queueMicrotask(() => child.onload()); },
        getContext: () => ({drawImage() {}}), toBlob: fn => fn({arrayBuffer: async () => new ArrayBuffer(150)}),
      };
      node.classList = {
        contains(name) {return node.className.split(' ').includes(name);},
        add(name) {node.className += ' ' + name;},
        remove(name) {node.className = node.className.split(' ').filter(x => x !== name).join(' ');},
        toggle(name, on) {if (on) this.add(name); else this.remove(name);},
      };
      return node;
    },
  },
  DOMParser: class {
    parseFromString() {
      return {body: {firstChild: null}, querySelector: selector => selector === '#overview' ? {} : {getAttribute: () => '/static/app.js'}, querySelectorAll: () => []};
    }
  },
  fetch: async (url, options = {}) => {
    calls.push({url, options});
    let data = {state: 'idle'};
    if (url.endsWith('/token')) data = {id: 'device-a', token: 'private-a'};
    if (url.endsWith('/browser-live') || url.endsWith('/status')) data = {state: 'running', kind: 'live camera', node_id: 'node-a', connected: true};
    if (url.endsWith('/cameras')) data = {cameras: [{node_id: 'node-a', status: 'online'}]};
    if (url.endsWith('/output')) data = {job: {}, incidents: []};
    if (url.endsWith('/stop')) data = {state: 'stopped'};
    return {ok: true, json: async () => data, text: async () => "<html>Overview</html>"};
  },
});
vm.runInContext(fs.readFileSync('snatchguard/central/static/footage.js', 'utf8'), context);
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  await element('#start-live').handlers.click();
  assert.equal(socketUrls[0], `${process.env.TEST_CAMERA_ORIGIN || 'wss://portal.example'}/api/v1/browsercam/ws?id=device-a&token=private-a`);
  const start = calls.find(c => c.url.endsWith('/browser-live'));
  assert.equal(start.options.headers['x-camera-id'], 'device-a');
  assert.equal(start.options.headers['x-camera-token'], 'private-a');
  assert.equal(JSON.parse(start.options.body).name, 'Gate');
  assert.equal(JSON.parse(start.options.body).site, 'Campus');
  await vm.runInContext('status()', context);
  assert.equal(element('#live-badge').textContent, '● LIVE · DETECTING');
  // Overview is displayed in this document: no pagehide, no WebSocket close, and frames keep flowing.
  await vm.runInContext('showOverview()', context);
  assert.equal(navigations.at(-1), '/#overview');
  assert.ok(hidden.has('hidden'));
  assert.equal(tracksStopped, 0);
  assert.ok(!calls.some(c => c.url.endsWith('/stop')));
  const before = sentFrames;
  timers.find(t => t.ms === 100).fn();
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(sentFrames > before, 'camera must keep sending frames on Overview');
  vm.runInContext('showFootage()', context);
  assert.equal(navigations.at(-1), '/footage');
  assert.ok(!hidden.has('hidden'));
  assert.equal(tracksStopped, 0);
  const html = fs.readFileSync('snatchguard/central/static/footage.html', 'utf8');
  const link = html.match(/<a[^>]*id="open-control-room"[^>]*>/)[0];
  assert.match(link, /href="\/#overview"/);
  assert.ok(!link.includes('target='));
  lifecycle.pagehide();
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(calls.some(c => c.url === '/api/v1/browsercam/stop' && c.options.keepalive && c.options.headers['x-camera-id'] === 'device-a'));
  assert.ok(!calls.some(c => c.url === '/api/v1/footage/stop'));
  assert.equal(tracksStopped, 1);
  element('#video').files = [{name: 'recording.mp4'}];
  await element('#upload-form').handlers.submit({preventDefault() {}});
  assert.ok(calls.some(c => c.url === '/api/v1/footage/upload'), 'recorded-video uploads must remain functional');
  let removed = false;
  let clip = '/media/incident/clip.mp4';
  let kind = 'uploaded video';
  context.fetch = async (url, options = {}) => {
    if (options.method === 'DELETE' || url.endsWith('/restore')) return {status: 404, ok: false, json: async () => ({detail: 'Not Found'})};
    if (url.endsWith('/review')) {
      removed = (JSON.parse(options.body).note || '').includes('[Deleted from portal]');
      return {ok: true, json: async () => ({})};
    }
    const event = {id: 'incident', clip, status: 'new', review_note: removed ? '[Deleted from portal]' : null};
    let data = event;
    if (url.includes('/footage/output')) data = {job: {id: 'job', kind, state: 'completed', name: 'clip.mp4'}, incidents: [{id: 'incident', clip, level: 'CONFIRMED'}]};
    if (url.includes('/events?')) data = [event];
    return {ok: true, json: async () => data};
  };
  await vm.runInContext('changeOutputVideo({id:"incident", clip_removed:false}, {disabled:false})', context);
  assert.ok(removed, 'old-server fallback must persist the removal');
  assert.equal(element('#output-list').children.length, 0);
  assert.match(element('#output-summary').textContent, /No snatch detected/);
  await vm.runInContext('output()', context); // Poll must honor the saved removal.
  assert.equal(vm.runInContext('legacyVideoState.get("incident")', context), true);
  await vm.runInContext('changeOutputVideo({id:"incident", clip_removed:true}, {disabled:false})', context);
  assert.equal(removed, false, 'old-server restore must clear the saved marker');
  assert.equal(element('#output-list').children[0].children.some(child => child.tag === 'video'), true);
  kind = 'real CCTV demo · training clip';
  await vm.runInContext('output()', context);
  assert.match(element('#output-summary').textContent, /Start the live camera or upload a video/);
  assert.equal(element('#output-list').children.length, 0, 'old demo must not appear as a current detection');
  console.log('Browser lifecycle passed: same-tab Overview, continued live frames, return navigation and scoped shutdown.');
})().catch(error => { console.error(error); process.exitCode = 1; });

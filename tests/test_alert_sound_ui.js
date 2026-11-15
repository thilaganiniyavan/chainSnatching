const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
let tones = 0;
function element(key) {
  if (!elements.has(key)) elements.set(key, {textContent: '', handlers: {}, addEventListener(name, callback) {this.handlers[name] = callback;}});
  return elements.get(key);
}
class Audio {
  constructor() {this.state = 'suspended'; this.currentTime = 0;}
  async resume() {this.state = 'running';}
  createOscillator() {tones++; return {frequency: {}, connect: gain => gain, start() {}, stop() {}};}
  createGain() {return {gain: {setValueAtTime() {}, exponentialRampToValueAtTime() {}}, connect() {}};}
}
const window = {AudioContext: Audio};
const context = vm.createContext({window, document: {querySelector: element, querySelectorAll: () => [], addEventListener() {}}, setInterval() {}, setTimeout() {}, console});
let source = fs.readFileSync('snatchguard/central/static/app.js', 'utf8');
source = source.replace('loadInitial().catch(() => {});', 'Promise.resolve().catch(() => {});').replace('\nconnect();', '');
source = source.replace(/\}\)\(\);\s*$/, 'window.test = {soundDetection, videoAction, changeVideo, state, isDismissed, passesFilter}; function renderFeed() {} function refreshStats() {}\n})();');
vm.runInContext(source, context);
(async () => {
  assert.match(element('#sound-btn').textContent, /Enable sound/);
  await element('#sound-btn').handlers.click();
  assert.match(element('#sound-btn').textContent, /Sound on/);
  tones = 0;
  window.test.soundDetection({id: 'new', level: 'CONFIRMED', type: 'snatch'});
  assert.equal(tones, 4);
  window.test.soundDetection({id: 'new', level: 'CONFIRMED', type: 'snatch'});
  assert.equal(tones, 4, 'review / duplicate messages must not sound twice');
  window.test.soundDetection({id: 'offline', level: 'OFFLINE', type: 'camera_offline'});
  assert.equal(tones, 4, 'offline state must not trigger a sound');
  await element('#sound-btn').handlers.click();
  window.test.soundDetection({id: 'muted', level: 'SUSPECT', type: 'snatch'});
  assert.equal(tones, 4);
  const action = window.test.videoAction({clip: '/media/a/clip.mp4'});
  assert.match(action, /aria-label="Move alert to Deleted"/);
  assert.match(action, />🗑<\/button>/);
  assert.match(window.test.videoAction({type: 'tamper'}), /data-video-action="delete"/);
  assert.match(window.test.videoAction({type: 'tamper'}), /aria-label="Move alert to Deleted"/);
  const requests = [];
  context.fetch = async (url, options = {}) => {
    requests.push({url, options});
    if (options.method === 'DELETE') return {status: 405, ok: false, json: async () => ({detail: 'Method Not Allowed'})};
    if (options.method !== 'POST') return {status: 200, ok: true, json: async () => ({id: 'old-server', type: 'tamper', status: 'new', review_note: 'Existing camera note'})};
    const note = JSON.parse(options.body).note;
    return {status: 200, ok: true, json: async () => ({id: 'old-server', type: 'tamper', payload: {}, review_note: note})};
  };
  window.test.state.events.set('old-server', {review_note: 'Existing camera note'});
  const button = {disabled: false};
  await window.test.changeVideo('old-server', 'delete', button);
  assert.equal(requests.at(-1).url, '/api/v1/events/old-server/review');
  assert.equal(JSON.parse(requests.at(-1).options.body).note, '[Deleted from portal] Existing camera note');
  assert.ok(window.test.state.events.has('old-server'));
  element('#search').value = '';
  const deleted = window.test.state.events.get('old-server');
  window.test.state.filter = 'all';
  assert.equal(window.test.passesFilter(deleted), false);
  assert.equal(window.test.videoAction(deleted), '');
  window.test.state.filter = 'deleted';
  assert.equal(window.test.passesFilter(deleted), true);
  assert.match(window.test.videoAction(deleted), /data-video-action="restore"/);
  assert.equal(window.test.passesFilter({id: 'active', type: 'snatch', payload: {}}), false);
  assert.ok(window.test.isDismissed({review_note: '[Deleted from portal] Existing camera note'}));
  console.log('Alert sound passed: first-click enable, immediate tone, duplicate suppression, mute and delete icon.');
})().catch(error => {console.error(error); process.exitCode = 1;});

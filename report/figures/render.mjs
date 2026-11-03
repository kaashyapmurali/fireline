// Capture the README figures from the website report page with local headless Chrome at 2x (light theme).
//   node report/figures/render.mjs [--url <report page>] [name ...]
// The website draws every figure from this repository's bundle (synced into the website), so the README shows the
// same figures in the same style. Run the website first (npm run dev in the website repo) or pass the deployed URL.
// Output: assets/*.{png,jpg}; assets/fire-timeline.gif is copied from the bundle (built by report/scripts/imagery.py).
// Loopback DevTools only. Adapted from the DisasterLens report pipeline.
import { spawn } from 'node:child_process';
import { copyFile, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
// README file name -> [website figure class (figure.fl.<class>), index among figures of that class, image format]
const FIGURES = {
  data: ['data-overview', 0, 'png'], 'model-size': ['model-size', 0, 'png'], 'training-loss': ['training-loss', 0, 'png'], architecture: ['arch', 0, 'jpeg'], explainer: ['explainer', 0, 'jpeg'],
  comparison: ['comparison', 0, 'png'], rings: ['rings', 0, 'png'], days: ['day-scatter', 0, 'png'],
  concentration: ['concentration', 0, 'png'], 'far-fire': ['far-fire', 0, 'png'], gap: ['gap', 0, 'jpeg'],
  'jepa-setup': ['jepa-setup', 0, 'png'], jepa: ['jepa', 0, 'png'], 'jepa-curves': ['curves', 0, 'png'],
  ladder: ['ladder', 0, 'png'], leak: ['leak', 0, 'png'], 'leak-fires': ['leak-fires', 0, 'png'],
  wind: ['wind', 0, 'png'], 'next-steps': ['next-steps', 0, 'png'], calibration: ['calibration', 0, 'png'],
};
const args = process.argv.slice(2);
const urlAt = args.indexOf('--url');
const url = urlAt >= 0 ? args[urlAt + 1] : 'http://localhost:4321/projects/fireline/';
const names = args.filter((_, i) => urlAt < 0 || (i !== urlAt && i !== urlAt + 1));
const wanted = names.length ? names : Object.keys(FIGURES);
const profile = await mkdtemp(join(tmpdir(), 'render-figures-'));
const browser = process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const local = /^https?:\/\/(localhost|127\.0\.0\.1)[:/]/.test(url);
const child = spawn(browser, ['--headless', '--hide-scrollbars', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--use-mock-keychain', '--password-store=basic', ...(local ? ['--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost'] : []), `--user-data-dir=${profile}`, '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const wait = ms => new Promise(r => setTimeout(r, ms));
let socket;
try {
  const endpoint = await new Promise((accept, reject) => {
    const timer = setTimeout(() => reject(new Error('Local Chrome startup timeout')), 15000);
    child.stderr.on('data', b => { const m = b.toString().match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/\S+)/); if (m) { clearTimeout(timer); accept(m[1]); } });
  });
  socket = new WebSocket(endpoint);
  await new Promise((a, r) => { socket.addEventListener('open', a, {once: true}); socket.addEventListener('error', r, {once: true}); });
  let sequence = 0;
  const requests = new Map(), errors = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description || 'exception');
    const pending = requests.get(message.id);
    if (pending) { requests.delete(message.id); message.error ? pending.reject(new Error(JSON.stringify(message.error))) : pending.resolve(message.result); }
  });
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++sequence; requests.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
  });
  const {targetId} = await call('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await call('Target.attachToTarget', {targetId, flatten: true});
  const page = (method, params) => call(method, params, sessionId);
  const evaluate = async expression => (await page('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true})).result.value;
  await page('Page.enable'); await page('Runtime.enable');
  await mkdir(join(root, 'assets'), {recursive: true});
  for (const name of wanted) if (name !== 'gif' && !FIGURES[name]) throw new Error(`Unknown figure ${name}`);
  await page('Emulation.setDeviceMetricsOverride', {width: 1280, height: 1000, deviceScaleFactor: 2, mobile: false});
  await page('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'light'}]});
  await page('Page.navigate', {url});
  await wait(1500);
  const ready = await evaluate(`(async () => {
    document.documentElement.setAttribute('data-theme', 'light');
    const style = document.createElement('style');
    style.textContent = 'header, #btt-btn-container, astro-dev-toolbar { display: none !important; } * { transition: none !important; animation: none !important; }';
    document.head.append(style);
    for (const img of document.images) img.loading = 'eager';
    await document.fonts.ready;
    await Promise.all([...document.images].map(i => i.decode().catch(() => null)));
    return document.querySelectorAll('figure.fl').length;
  })()`);
  if (!ready) throw new Error(`No report figures found at ${url}; is the website running?`);
  await wait(500);
  const missing = await evaluate('[...document.images].filter(i => i.getAttribute("src") && !i.naturalWidth).map(i => i.getAttribute("src"))');
  if (missing.length) throw new Error(`Images failed to load: ${missing.join(', ')}`);
  for (const name of wanted.filter(n => n !== 'gif')) {
    const [selector, index, format] = FIGURES[name];
    const file = join(root, 'assets', `${name}.${format === 'jpeg' ? 'jpg' : 'png'}`);
    const box = await evaluate(`(() => { const r = document.querySelectorAll('figure.fl.${selector}')[${index}].getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: Math.ceil(r.width), height: Math.ceil(r.height)}; })()`);
    const pad = 12;  // keep a margin of the page background around the figure
    const clip = {x: box.x - pad, y: box.y - pad, width: box.width + 2 * pad, height: box.height + 2 * pad, scale: 1};
    const {data} = await page('Page.captureScreenshot', {format, quality: format === 'jpeg' ? 88 : undefined, clip, captureBeyondViewport: true});
    await writeFile(file, Buffer.from(data, 'base64'));
    console.log(`${name}: ${clip.width}x${clip.height} css px -> ${file.replace(root + '/', '')}`);
  }
  if (!names.length || names.includes('gif')) {
    await copyFile(join(root, 'report/bundle/images/timeline.gif'), join(root, 'assets/fire-timeline.gif'));
    console.log('gif: report/bundle/images/timeline.gif -> assets/fire-timeline.gif');
  }
  if (errors.length) throw new Error(`JavaScript errors: ${errors.join(' | ')}`);
  await call('Browser.close');
} finally {
  if (socket) socket.close();
  child.kill();
  await wait(500);
  await rm(profile, {recursive: true, force: true});
}

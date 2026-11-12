// Viewer page: the composite of all visible channels plus the mask, frame navigation and playback (desktop frame
// rate from utils/config.py), Z-projections, cursor read-out, ROI drawing, histogram and PNG export.
import { route, app, esc, onLeave, download } from './core.js';
import { S, ensureData, onChange, frameCount, dims, frameOf } from './session.js';
import { paint, drawHistogram, maskFrame, fmt } from './render.js';
import { panelHTML, renderPanel, renderRois, bindLoaders } from './viewer_panel.js';
import { run } from './tasks.js';

const projCache = new Map();  // `${channel id}:${method}` -> projected plane

route('viewer', async () => {
  await ensureData();
  app().innerHTML = `
  <h1>Viewer</h1>
  <p class="lead">Channels are mapped through their LUTs between min and max and added together; the mask is mixed in
  with the desktop app's overlay formula. Everything runs in this browser tab.</p>
  <div class="side viewer">
    <div>${panelHTML()}</div>
    <div>
      <div class="stage" id="v-stage"><canvas id="v-canvas" role="img" aria-label="Composite of the loaded channels and mask"></canvas></div>
      <div class="navrow">
        <button type="button" id="v-prev" class="alt sm" aria-label="Previous frame">◀</button>
        <button type="button" id="v-play" class="sm">Play</button>
        <button type="button" id="v-next" class="alt sm" aria-label="Next frame">▶</button>
        <div class="grow"><label for="v-frame">Frame <b id="v-fl"></b></label><input type="range" id="v-frame" min="0" value="0" step="1"></div>
      </div>
      <p class="mono small" id="v-read" aria-live="polite">Move the pointer over the image.</p>
      <div class="panel"><h3 id="v-hh">Histogram</h3><canvas id="v-hist" width="600" height="110" class="hist" role="img" aria-label="Intensity histogram of the selected channel"></canvas></div>
      <button type="button" id="v-png" class="alt">Save view as PNG</button>
      <p class="small mute" id="v-info"></p>
    </div>
  </div>`;
  const root = app(), canvas = root.querySelector('#v-canvas'), slider = root.querySelector('#v-frame');
  const say = (msg, bad = false) => { const m = root.querySelector('#v-msg'); m.textContent = msg; m.className = 'small ' + (bad ? 'err' : 'mute'); };
  let draft = null, timer = null;

  const shown = ch => (S.view === 'frame' ? frameOf(ch, S.frame) : projCache.get(`${ch.id}:${S.view}`) || frameOf(ch, S.frame));
  async function ensureProjections() {
    if (S.view === 'frame') return;
    for (const ch of S.channels.filter(c => c.visible)) {
      const key = `${ch.id}:${S.view}`;
      if (!projCache.has(key)) projCache.set(key, (await run('projection', ch.frames, S.view, ch.dtype)).data);
    }
  }
  function redraw() {
    const d = dims();
    if (!d) { canvas.width = canvas.height = 1; return; }
    const n = frameCount();
    slider.max = n - 1; slider.value = S.frame; slider.disabled = S.view !== 'frame';
    root.querySelector('#v-fl').textContent = S.view === 'frame' ? `${S.frame + 1} / ${n}` : `${S.view} projection of ${n} frames`;
    const scaleCh = S.channels.find(c => c.scaleKnown);
    paint(canvas, { channels: S.channels.filter(c => c.visible), planesFor: shown, W: d.W, H: d.H,
      mask: S.view === 'frame' ? maskFrame(S.frame) : null, alpha: S.maskAlpha, color: S.maskColor,
      scale: scaleCh, rois: S.rois, draft });
    const sel = S.channels[S.selected];
    if (sel) { drawHistogram(root.querySelector('#v-hist'), shown(sel), sel.min, sel.max); root.querySelector('#v-hh').textContent = `Histogram: ${sel.name}`; }
    root.querySelector('#v-info').textContent = S.channels.map(c => `${c.name}: ${c.layout}; scale ${c.scaleSource}`).join(' · ');
  }
  redraw.shown = shown;
  const refresh = async () => { await ensureProjections(); renderPanel(root, redraw); redraw(); };

  const files = bindLoaders(root, say);
  const off = onChange(what => { if (what === 'rois') { renderRois(root); redraw(); } else { projCache.clear(); refresh(); } });
  const setFrame = t => { const n = frameCount(); S.frame = ((t % n) + n) % n; redraw(); };
  slider.addEventListener('input', () => setFrame(+slider.value));
  root.querySelector('#v-prev').addEventListener('click', () => setFrame(S.frame - 1));
  root.querySelector('#v-next').addEventListener('click', () => setFrame(S.frame + 1));
  const play = root.querySelector('#v-play');
  const stop = () => { clearInterval(timer); timer = null; play.textContent = 'Play'; };
  play.addEventListener('click', () => {
    if (timer) return stop();
    play.textContent = 'Pause';
    timer = setInterval(() => {
      if (!S.cfg.loop_playback && S.frame === frameCount() - 1) return stop();
      setFrame(S.frame + 1);
    }, 1000 / S.cfg.frame_rate);
  });
  root.querySelector('#v-view').value = S.view;
  root.querySelector('#v-view').addEventListener('change', async e => { S.view = e.target.value; say(S.view === 'frame' ? '' : 'Projecting…'); await refresh(); say(''); });
  root.querySelector('#v-png').addEventListener('click', () => canvas.toBlob(b => download(`overlay_${S.view === 'frame' ? 'frame' + (S.frame + 1) : S.view}.png`, b)));

  const stage = root.querySelector('#v-stage');
  stage.addEventListener('dragover', e => { e.preventDefault(); stage.classList.add('drop'); });
  stage.addEventListener('dragleave', () => stage.classList.remove('drop'));
  stage.addEventListener('drop', e => { e.preventDefault(); stage.classList.remove('drop'); files([...e.dataTransfer.files]); });

  // pointer: read-out and ROI drawing. Coordinates are 0-based pixel indices (column x, row y) from the top-left.
  const toPx = e => { const r = canvas.getBoundingClientRect(), d = dims(); return [(e.clientY - r.top) * d.H / r.height, (e.clientX - r.left) * d.W / r.width]; };
  let start = null;
  canvas.addEventListener('pointerdown', e => {
    const tool = root.querySelector('#v-tool').value;
    if (!tool) return;
    start = toPx(e); canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    const [y, x] = toPx(e), d = dims();
    if (!d) return;
    const xi = Math.min(d.W - 1, Math.max(0, Math.floor(x))), yi = Math.min(d.H - 1, Math.max(0, Math.floor(y))), i = yi * d.W + xi;
    const sc = S.channels.find(c => c.scaleKnown);
    const vals = S.channels.map(c => `${esc(c.name)} ${fmt(shown(c)[i])}`).join(' · ');
    const m = S.mask && S.view === 'frame' ? ` · mask ${frameOf(S.mask, S.frame)[i] > 0 ? 'in' : 'out'}` : '';
    root.querySelector('#v-read').innerHTML = `x ${xi}, y ${yi}${sc ? ` (${(xi * sc.um_per_px).toFixed(2)}, ${(yi * sc.um_per_px).toFixed(2)} µm)` : ''} · ${vals}${m}`;
    if (!start) return;
    const tool = root.querySelector('#v-tool').value;
    draft = tool === 'rectangle'
      ? { type: 'rectangle', points: [Math.round(Math.min(start[0], y)), Math.round(Math.min(start[1], x)), Math.round(Math.max(start[0], y)), Math.round(Math.max(start[1], x))] }
      : { type: 'circle', points: [start[0], start[1], Math.hypot(y - start[0], x - start[1])] };
    redraw();
  });
  canvas.addEventListener('pointerup', () => {
    if (draft && (draft.type === 'circle' ? draft.points[2] >= 1 : draft.points[2] > draft.points[0] && draft.points[3] > draft.points[1])) {
      draft.label = `R${S.rois.length + 1}`; S.rois.push(draft);
      draft = null; start = null; renderRois(root); redraw();
    } else { draft = null; start = null; }
  });
  const keys = e => {
    if (e.target.closest('input, select, textarea')) return;
    if (e.key === 'ArrowRight') setFrame(S.frame + 1); else if (e.key === 'ArrowLeft') setFrame(S.frame - 1);
  };
  document.addEventListener('keydown', keys);
  onLeave(() => { stop(); off(); document.removeEventListener('keydown', keys); });
  await refresh();
});

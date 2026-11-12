// The loaded data, shared by the viewer, process and analyze pages (kept in memory for this browser tab only).
// channels: fluorescence stacks, each { id, name, file, T, H, W, dtype, frames[] (typed arrays), lut, min, max,
// visible, um_per_px, scaleKnown, scaleSource, layout }. mask: one binary stack (> 0 is inside), as the desktop app's
// "mask stack". All stacks must share H x W; the frame count may differ (the shorter stack holds its last frame).
import { api } from './core.js';
import { readTiff } from './tiff_reader.js';
import { minMax, percentile } from './ops.js';

export const S = { demo: false, channels: [], mask: null, frame: 0, rois: [], view: 'frame', selected: 0, cfg: null, luts: null, listeners: new Set() };
let nextId = 1;
const LUT_CYCLE = ['Green', 'Magenta', 'Cyan', 'Yellow', 'Red', 'Blue', 'Grays'];

export async function config() {
  if (!S.cfg) {
    const r = await api('/luts');
    S.cfg = r.display; S.luts = r.luts; S.maskColors = r.mask_colors; S.desktopColormaps = r.desktop_colormaps;
    S.maskAlpha = r.display.overlay_alpha; S.maskColor = r.display.mask_color;
  }
  return S;
}

export const changed = what => S.listeners.forEach(fn => fn(what));
export function onChange(fn) { S.listeners.add(fn); return () => S.listeners.delete(fn); }

export const frameCount = () => Math.max(1, ...S.channels.map(c => c.T), S.mask ? S.mask.T : 1);
export const dims = () => (S.channels[0] || S.mask || null);
export const frameOf = (stack, t) => stack.frames[Math.min(t, stack.T - 1)];

function checkSize(H, W, name) {
  const d = dims();
  if (d && (d.H !== H || d.W !== W)) throw new Error(`${name} is ${W} x ${H} pixels, but the loaded stacks are ${d.W} x ${d.H}. Clear them first.`);
}

// Auto levels: 1st-99th percentile of one frame (gui/image_view.py _get_display_levels)
export function autoLevels(ch, data = frameOf(ch, S.frame)) {
  const f = data, [p1, p2] = S.cfg ? S.cfg.auto_levels_percentiles : [1, 99];
  let lo = percentile(f, p1), hi = percentile(f, p2);
  if (hi <= lo) [lo, hi] = minMax(f);
  if (hi <= lo) hi = lo + 1;
  ch.min = lo; ch.max = hi;
}

export function addChannel({ name, file = '', frames, T, H, W, dtype, lut, meta = {} }) {
  checkSize(H, W, name);
  const ch = { id: nextId++, name, file, frames, T, H, W, dtype, visible: true,
    lut: lut || LUT_CYCLE[S.channels.length % LUT_CYCLE.length], min: 0, max: 1,
    um_per_px: meta.um_per_px ?? 1, scaleKnown: !!meta.scaleKnown, scaleSource: meta.scaleSource || 'none (pixels)',
    frame_s: meta.frame_s || 0, layout: meta.layout || '' };
  let lo = Infinity, hi = -Infinity;
  for (const f of frames) { const [a, b] = minMax(f); if (a < lo) lo = a; if (b > hi) hi = b; }
  ch.lo = lo; ch.hi = hi > lo ? hi : lo + 1;  // whole-stack range: the limits of the contrast sliders
  autoLevels(ch, frames[0]);
  S.channels.push(ch);
  if (S.channels.length === 1) S.frame = 0;
  changed('channels');
  return ch;
}

export function setMask({ name, frames, T, H, W, file = '' }) {
  checkSize(H, W, name);
  S.mask = { name, file, frames, T, H, W, visible: true };
  changed('mask');
}

export function removeChannel(id) {
  S.channels = S.channels.filter(c => c.id !== id);
  S.selected = Math.min(S.selected, Math.max(0, S.channels.length - 1));
  changed('channels');
}

export function clearAll() { S.demo = false; S.channels = []; S.mask = null; S.rois = []; S.frame = 0; S.view = 'frame'; changed('channels'); }

// Split a reader result into frames per channel: T x Z planes become frames (z fastest within each time point)
export function splitStack(st) {
  return Array.from({ length: st.C }, (_, c) => {
    const frames = [];
    for (let t = 0; t < st.T; t++) for (let z = 0; z < st.Z; z++) frames.push(st.plane(t, z, c));
    return frames;
  });
}

// Load one TIFF (ArrayBuffer) as channels or as the mask (its first channel)
export async function loadTiff(buf, file, role = 'channel', opts = {}) {
  const st = await readTiff(buf);
  if (S.demo && !opts.demo) { clearAll(); S.demo = false; }  // a user's own file replaces the demo
  const parts = splitStack(st), T = st.T * st.Z;
  const meta = { um_per_px: st.um_per_px, scaleKnown: st.scaleKnown, scaleSource: st.scaleSource, frame_s: st.frame_s,
    layout: `${st.layout}; ${st.pages} pages, ${st.dtype}${st.Z > 1 ? `, ${st.T} time points x ${st.Z} slices as frames` : ''}` };
  if (role === 'mask') {
    setMask({ name: opts.name || file, file, frames: parts[0], T, H: st.H, W: st.W });
    return { st, added: ['mask'] };
  }
  const added = parts.map((frames, c) => addChannel({ name: (opts.name || file) + (st.C > 1 ? ` [c${c + 1}]` : ''), file,
    frames, T, H: st.H, W: st.W, dtype: st.dtype, lut: st.C > 1 ? undefined : opts.lut, meta }).name);
  return { st, added };
}

export async function loadDemo() {
  await config();
  const { demos } = await api('/demos');
  clearAll();
  for (const d of demos) {
    const r = await fetch('static/' + d.file);
    if (!r.ok) throw new Error(`${r.status} loading ${d.file}`);
    await loadTiff(await r.arrayBuffer(), d.file.split('/').pop(), d.role, { name: d.name, lut: d.lut, demo: true });
  }
  S.demo = true;
  changed('demo');
}

// First visit to a page with nothing loaded: open the demo so every page shows something
export async function ensureData() {
  await config();
  if (!S.channels.length && !S.mask) await loadDemo();
}

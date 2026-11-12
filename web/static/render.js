// Drawing the view: visible channels mapped through their LUTs between min and max and added together (a
// "composite", clamped at 255), then the mask mixed in with the desktop app's formula (core/overlay_blend.py):
// out = base * (1 - alpha * m) + colour * alpha * m. Also the scale bar, ROIs and the histogram canvas.
import { S, frameOf } from './session.js';

// planesFor(ch) returns the 2D data to show for a channel (the current frame or a projection)
export function compositeRGBA(channels, planesFor, W, H, { mask = null, alpha = 0.5, color = [255, 0, 0] } = {}) {
  const rgb = new Float32Array(W * H * 3);
  for (const ch of channels) {
    const lut = S.luts[ch.lut] || S.luts.Grays, data = planesFor(ch), lo = ch.min, span = ch.max - ch.min || 1;
    for (let i = 0; i < W * H; i++) {
      let k = Math.round(((data[i] - lo) / span) * 255);
      k = k < 0 ? 0 : k > 255 ? 255 : k;
      const c = lut[k];
      rgb[i * 3] += c[0]; rgb[i * 3 + 1] += c[1]; rgb[i * 3 + 2] += c[2];
    }
  }
  const img = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const m = mask && mask[i] > 0 ? 1 : 0;
    for (let c = 0; c < 3; c++) {
      const base = Math.min(255, rgb[i * 3 + c]);
      img.data[i * 4 + c] = m ? Math.trunc(Math.trunc(base) * (1 - alpha * m) + color[c] * alpha * m) : base;
    }
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

// Pick a "nice" scale bar length (1, 2 or 5 x 10^n um) about a fifth of the image width
export function niceBar(widthUm) {
  const target = widthUm / 5, p = 10 ** Math.floor(Math.log10(target));
  return [1, 2, 5, 10].map(m => m * p).reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
}

// Overlays are drawn in image pixel units; the context is already scaled by the display factor
export function drawOverlays(ctx, W, H, { scale = null, rois = [], draft = null, points = [], k = 1 } = {}) {
  const s = Math.max(W, H) / 300;  // line width / text unit: a fixed fraction of the image
  ctx.lineWidth = s;
  for (const r of [...rois, ...(draft ? [draft] : [])]) {
    ctx.strokeStyle = r === draft ? '#ffffff' : '#ffd23f';
    ctx.beginPath();
    const p = r.points;
    if (r.type === 'rectangle') ctx.rect(p[1], p[0], p[3] - p[1], p[2] - p[0]);
    else ctx.arc(p[1], p[0], Math.max(0.5, p[2]), 0, 2 * Math.PI);
    ctx.stroke();
    if (r.label) { ctx.fillStyle = '#ffd23f'; ctx.font = `${10 * s}px monospace`; ctx.fillText(r.label, p[1] + 2 * s, p[0] - 2 * s); }
  }
  ctx.strokeStyle = '#3dff8f';
  for (const [y, x, r] of points) { ctx.beginPath(); ctx.arc(x, y, Math.max(1.5 * s, (r || 1) * Math.SQRT2), 0, 2 * Math.PI); ctx.stroke(); }
  if (scale && scale.scaleKnown) {
    const um = niceBar(W * scale.um_per_px), px = um / scale.um_per_px, y = H - 8 * s;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(W - px - 8 * s, y, px, 3 * s);
    ctx.font = `${10 * s}px monospace`; ctx.textAlign = 'right';
    ctx.fillText(`${um} µm`, W - 8 * s, y - 4 * s); ctx.textAlign = 'left';
  }
}

// Paint a full view: the image at native size, enlarged by an integer factor k (pixels stay square and sharp) so
// that text and outlines drawn on top are crisp. canvas.dataset.k records k for pointer mapping.
export function paint(canvas, { channels, planesFor, W, H, mask, alpha, color, scale, rois, draft, points }) {
  const k = Math.max(1, Math.min(8, Math.floor(1024 / Math.max(W, H))));
  const src = document.createElement('canvas');
  src.width = W; src.height = H;
  src.getContext('2d').putImageData(compositeRGBA(channels, planesFor, W, H, { mask, alpha, color }), 0, 0);
  canvas.width = W * k; canvas.height = H * k; canvas.dataset.k = k;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, W * k, H * k);
  ctx.setTransform(k, 0, 0, k, 0, 0);
  drawOverlays(ctx, W, H, { scale, rois, draft, points, k });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// Histogram (64 bins between the frame's min and max) with the display range marked
export function drawHistogram(canvas, data, lo, hi, color = '#4f9dff') {
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height, bins = new Float64Array(64);
  let a = Infinity, b = -Infinity;
  for (const v of data) { if (v < a) a = v; if (v > b) b = v; }
  const span = b - a || 1;
  for (const v of data) bins[Math.min(63, Math.floor(((v - a) / span) * 64))]++;
  const top = Math.log1p(Math.max(...bins));
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = color;
  bins.forEach((n, i) => { const bh = (Math.log1p(n) / top) * (h - 14); ctx.fillRect((i * w) / 64, h - 12 - bh, w / 64 - 1, bh); });
  ctx.strokeStyle = '#ffd23f';
  for (const v of [lo, hi]) { const x = ((v - a) / span) * w; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - 12); ctx.stroke(); }
  ctx.fillStyle = '#94a3c6'; ctx.font = '10px monospace';
  ctx.fillText(fmt(a), 2, h - 2); ctx.textAlign = 'right'; ctx.fillText(fmt(b), w - 2, h - 2); ctx.textAlign = 'left';
}

export const fmt = v => (Math.abs(v) >= 1000 || Number.isInteger(v) ? Math.round(v).toLocaleString('en') : (+v).toPrecision(4));

export const maskFrame = t => (S.mask && S.mask.visible ? frameOf(S.mask, t) : null);

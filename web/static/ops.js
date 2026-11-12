// JavaScript ports of the repository's stack operations (core/image_loader.py, core/image_processor.py,
// core/overlay_blend.py), checked against the Python code by ops.test.mjs:
//   projection       ImageStack.create_z_projection (max, min, mean, median, std)
//   normalizeFrame   ImageStack.get_normalized_frame (min-max to 0-255)
//   applyContrast    ImageStack.apply_contrast: stack percentiles -> skimage rescale_intensity to the type's range
//   rescaleFrame     ImageProcessor.enhance_contrast('rescale'): frame percentiles -> frame's own min..max (float)
//   equalizeHist     ImageProcessor.enhance_contrast('equalize') = skimage.exposure.equalize_hist
//   otsu, segment    ImageProcessor.segment_image('threshold') = image > skimage threshold_otsu
//   measure          ImageProcessor.measure_intensity (whole frame, mask, rectangle or circle regions)
//   timeSeries       ImageProcessor.extract_time_series; basicStats: DataAnalyzer.calculate_basic_stats
//   blendOverlay     core/overlay_blend.blend_overlay (the desktop's mask-over-fluorescence composite)
import { ARRAY, INT_MAX, isFloat, castTo } from './filters.js';

export function minMax(a) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

// numpy.percentile, linear method, on an already sorted Float64Array
export function percentileSorted(s, p) {
  const pos = (p / 100) * (s.length - 1), i = Math.floor(pos), t = pos - i;
  if (i + 1 >= s.length) return s[s.length - 1];
  const a = s[i], b = s[i + 1], d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;  // numpy's _lerp
}
export const percentile = (a, p) => percentileSorted(Float64Array.from(a).sort(), p);

// frames: array of flat frames (T x H*W). Max/min keep the type; mean/median/std are float64.
export function projection(frames, method, dtype) {
  const n = frames[0].length, T = frames.length;
  if (method === 'max' || method === 'min') {
    const out = new ARRAY[dtype](frames[0]);
    for (const f of frames) for (let i = 0; i < n; i++) out[i] = method === 'max' ? Math.max(out[i], f[i]) : Math.min(out[i], f[i]);
    return { data: out, dtype };
  }
  const out = new Float64Array(n), col = new Float64Array(T);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < T; t++) col[t] = frames[t][i];
    if (method === 'median') { col.sort(); out[i] = T % 2 ? col[(T - 1) / 2] : (col[T / 2 - 1] + col[T / 2]) / 2; continue; }
    let s = 0;
    for (let t = 0; t < T; t++) s += col[t];
    const m = s / T;
    if (method === 'mean') { out[i] = m; continue; }
    let v = 0;
    for (let t = 0; t < T; t++) v += (col[t] - m) ** 2;
    out[i] = Math.sqrt(v / T);
  }
  return { data: out, dtype: 'float64' };
}

export function normalizeFrame(frame, range = null) {
  let [lo, hi] = range || minMax(frame);
  if (hi === lo) hi = lo + 1;
  const out = new Uint8Array(frame.length);
  for (let i = 0; i < frame.length; i++) out[i] = Math.trunc((frame[i] - lo) / (hi - lo) * 255);
  return out;
}

// skimage DTYPE_RANGE with clip_negative (the input minimum is >= 0 for microscopy data)
export function dtypeRange(dtype, clipNegative) {
  if (isFloat(dtype)) return [clipNegative ? 0 : -1, 1];
  const hi = INT_MAX[dtype], lo = dtype.startsWith('u') ? 0 : -hi - 1;
  return [clipNegative ? 0 : lo, hi];
}

function rescale(values, imin, imax, omin, omax) {
  const out = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = Math.min(Math.max(values[i], imin), imax);
    out[i] = imin !== imax ? ((v - imin) / (imax - imin)) * (omax - omin) + omin : Math.min(Math.max(v, omin), omax);
  }
  return out;
}

// Whole-stack contrast stretch (returns new frames in the same type)
export function applyContrast(frames, dtype, pLow, pHigh) {
  const all = new Float64Array(frames.length * frames[0].length);
  frames.forEach((f, t) => all.set(f, t * f.length));
  all.sort();
  const lo = percentileSorted(all, pLow), hi = percentileSorted(all, pHigh), [omin, omax] = dtypeRange(dtype, lo >= 0);
  return frames.map(f => castTo(rescale(f, lo, hi, omin, omax), dtype));
}

export function rescaleFrame(frame, pLow, pHigh) {
  const s = Float64Array.from(frame).sort(), [lo, hi] = [s[0], s[s.length - 1]];
  return rescale(frame, percentileSorted(s, pLow), percentileSorted(s, pHigh), lo, hi);
}

// skimage.exposure.histogram(source_range='image'): one bin per integer for integer types, else 256 equal bins
export function histogram(frame, dtype, nbins = 256) {
  const [lo, hi] = minMax(frame);
  if (!isFloat(dtype)) {
    const counts = new Float64Array(hi - lo + 1), centers = new Float64Array(hi - lo + 1);
    for (let i = 0; i < frame.length; i++) counts[frame[i] - lo]++;
    for (let i = 0; i < centers.length; i++) centers[i] = lo + i;
    return { counts, centers };
  }
  const counts = new Float64Array(nbins), centers = new Float64Array(nbins), w = (hi - lo) / nbins;
  for (let i = 0; i < nbins; i++) centers[i] = (lo + w * i + (lo + w * (i + 1))) / 2;
  for (let i = 0; i < frame.length; i++) {
    let b = w > 0 ? Math.floor(((frame[i] - lo) / (hi - lo)) * nbins) : 0;
    if (b >= nbins) b = nbins - 1;
    counts[b]++;
  }
  return { counts, centers };
}

export function equalizeHist(frame, dtype) {
  const { counts, centers } = histogram(frame, dtype), cdf = new Float64Array(counts.length);
  let s = 0;
  for (let i = 0; i < counts.length; i++) { s += counts[i]; cdf[i] = s; }
  const f32 = dtype === 'float32';
  for (let i = 0; i < cdf.length; i++) cdf[i] = f32 ? Math.fround(cdf[i] / s) : cdf[i] / s;
  const out = f32 ? new Float32Array(frame.length) : new Float64Array(frame.length), n = centers.length;
  for (let i = 0; i < frame.length; i++) {  // numpy.interp
    const v = frame[i];
    if (v <= centers[0]) { out[i] = cdf[0]; continue; }
    if (v >= centers[n - 1]) { out[i] = cdf[n - 1]; continue; }
    let a = 0, b = n - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (centers[m] <= v) a = m; else b = m; }
    out[i] = cdf[a] + (v - centers[a]) * (cdf[b] - cdf[a]) / (centers[b] - centers[a]);
  }
  return out;
}

export function otsu(frame, dtype) {
  const [lo, hi] = minMax(frame);
  if (lo === hi) return lo;
  const { counts, centers } = histogram(frame, dtype), n = counts.length;
  const w1 = new Float64Array(n), m1 = new Float64Array(n), w2 = new Float64Array(n), m2 = new Float64Array(n);
  let cw = 0, cm = 0;
  for (let i = 0; i < n; i++) { cw += counts[i]; cm += counts[i] * centers[i]; w1[i] = cw; m1[i] = cm / cw; }
  cw = 0; cm = 0;
  for (let i = n - 1; i >= 0; i--) { cw += counts[i]; cm += counts[i] * centers[i]; w2[i] = cw; m2[i] = cm / cw; }
  let best = -1, idx = 0;
  for (let i = 0; i < n - 1; i++) {
    const v = w1[i] * w2[i + 1] * (m1[i] - m2[i + 1]) ** 2;
    if (v > best) { best = v; idx = i; }
  }
  return centers[idx];
}

export function segment(frame, dtype, threshold = null) {
  const t = threshold ?? otsu(frame, dtype), out = new Uint8Array(frame.length);
  for (let i = 0; i < frame.length; i++) out[i] = frame[i] > t ? 1 : 0;
  return { mask: out, threshold: t };
}

function stats(values) {
  const n = values.length;
  if (!n) return null;
  let s = 0, lo = Infinity, hi = -Infinity;
  for (const v of values) { s += v; if (v < lo) lo = v; if (v > hi) hi = v; }
  const m = s / n;
  let q = 0;
  for (const v of values) q += (v - m) ** 2;
  return { mean: m, std: Math.sqrt(q / n), min: lo, max: hi, sum: s, count: n };
}

// region: { type: 'rectangle', points: [y1, x1, y2, x2] } or { type: 'circle', points: [y, x, r] }
export function regionMask(region, H, W) {
  const m = new Uint8Array(H * W), p = region.points;
  if (region.type === 'rectangle') {
    for (let y = Math.max(0, p[0]); y < Math.min(H, p[2]); y++) for (let x = Math.max(0, p[1]); x < Math.min(W, p[3]); x++) m[y * W + x] = 1;
  } else if (region.type === 'circle') {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (Math.sqrt((x - p[1]) ** 2 + (y - p[0]) ** 2) <= p[2]) m[y * W + x] = 1;
  } else throw new Error(`Unsupported region type: ${region.type}`);
  return m;
}

export function measure(frame, mask = null) {
  if (!mask) return stats(frame);
  const vals = [];
  for (let i = 0; i < frame.length; i++) if (mask[i] > 0) vals.push(frame[i]);
  return stats(vals);
}

export function timeSeries(frames, mask = null) {
  const out = { mean: [], std: [], sum: [], max: [], min: [] };
  for (const f of frames) {
    const s = measure(f, mask);
    for (const k of Object.keys(out)) out[k].push(s ? s[k] : null);
  }
  return out;
}

export function basicStats(frames) {
  const out = { mean: [], median: [], std: [], min: [], max: [], snr: [] };
  for (const f of frames) {
    const s = stats(f), sorted = Float64Array.from(f).sort(), n = sorted.length;
    out.mean.push(s.mean); out.std.push(s.std); out.min.push(s.min); out.max.push(s.max);
    out.median.push(n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2);
    out.snr.push(s.std > 0 ? s.mean / s.std : 0);
  }
  return out;
}

// RGB uint8 (H*W*3): frame scaled to 8 bits by its min/max, mask pixels mixed with color at opacity alpha
export function blendOverlay(frame, mask, alpha, color, dtype) {
  const base = dtype === 'uint8' ? frame : normalizeFrame(frame), out = new Uint8Array(frame.length * 3);
  for (let i = 0; i < frame.length; i++) {
    const m = mask[i] > 0 ? 1 : 0;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.trunc(base[i] * (1 - alpha * m) + color[c] * alpha * m);
  }
  return out;
}

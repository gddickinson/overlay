// JavaScript ports of ImageProcessor.apply_filter (core/image_processor.py), which calls scikit-image / SciPy:
//   gaussian  : skimage.filters.gaussian(preserve_range=True) = scipy.ndimage.gaussian_filter, mode 'nearest',
//               truncate 4, then cast back to the input type (integers truncate, as numpy's astype does)
//   bilateral : the repository code runs the same Gaussian with sigma = sigma_spatial (not a true bilateral filter)
//   median    : skimage.filters.median(footprint=disk(size)) = scipy.ndimage.median_filter, mode 'nearest'
//   sobel     : skimage.filters.sobel: image as float (integers scaled to 0-1 by their type's maximum), edge
//               magnitude sqrt((Gy^2 + Gx^2) / 2) with [1, 0, -1] x [1, 2, 1] / 4 kernels, mode 'reflect'
//   laplacian : skimage.filters.laplace: 3 x 3 Laplacian [0 -1 0; -1 4 -1; 0 -1 0] on the float image, mode 'reflect'
// Frames are flat arrays (row-major, H x W) of any numeric type. Checked against Python by ops.test.mjs.

export const INT_MAX = { uint8: 255, uint16: 65535, uint32: 4294967295, int8: 127, int16: 32767, int32: 2147483647 };
export const ARRAY = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array,
  int32: Int32Array, float32: Float32Array, float64: Float64Array };
export const isFloat = dtype => dtype.startsWith('float');

// numpy astype from float: integers truncate toward zero (values are kept in range by the callers' inputs)
export function castTo(values, dtype) {
  const out = new ARRAY[dtype](values.length);
  for (let i = 0; i < values.length; i++) out[i] = isFloat(dtype) ? values[i] : Math.trunc(values[i]);
  return out;
}

// skimage img_as_float: float input stays, unsigned integers are divided by their type's maximum, signed by theirs
export function asFloat(frame, dtype) {
  const out = new Float64Array(frame.length);
  if (isFloat(dtype)) { out.set(frame); return out; }
  const m = INT_MAX[dtype];
  for (let i = 0; i < frame.length; i++) out[i] = Math.max(frame[i] / m, -1);
  return out;
}

const clampIdx = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
const reflectIdx = (i, n) => {  // scipy 'reflect': d c b a | a b c d | d c b a
  const p = 2 * n;
  i = ((i % p) + p) % p;
  return i < n ? i : p - 1 - i;
};

// 1D correlation along rows (axis 1) or columns (axis 0); `round32` mimics float32 storage between passes
function correlate1d(src, H, W, weights, axis, edge, round32 = false) {
  const r = (weights.length - 1) / 2, out = new Float64Array(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) {
        const v = axis === 1 ? src[y * W + edge(x + k, W)] : src[edge(y + k, H) * W + x];
        s += weights[k + r] * v;
      }
      out[y * W + x] = round32 ? Math.fround(s) : s;
    }
  }
  return out;
}

export function gaussianKernel(sigma, truncate = 4) {
  const r = Math.trunc(truncate * sigma + 0.5), w = [];
  let sum = 0;
  for (let x = -r; x <= r; x++) { const v = Math.exp(-0.5 * x * x / (sigma * sigma)); w.push(v); sum += v; }
  return w.map(v => v / sum);
}

export function gaussian(frame, H, W, dtype, sigma) {
  const f32 = dtype === 'float32', k = gaussianKernel(sigma);
  const a = correlate1d(Float64Array.from(frame), H, W, k, 0, clampIdx, f32);
  return castTo(correlate1d(a, H, W, k, 1, clampIdx, f32), dtype);
}

export function disk(radius) {
  const r = Math.trunc(radius), pts = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= radius * radius) pts.push([dy, dx]);
  return pts;
}

export function median(frame, H, W, dtype, size) {
  const pts = disk(size), out = new ARRAY[dtype](frame.length), buf = new Float64Array(pts.length);
  const rank = Math.floor(pts.length / 2);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      for (let i = 0; i < pts.length; i++) buf[i] = frame[clampIdx(y + pts[i][0], H) * W + clampIdx(x + pts[i][1], W)];
      buf.sort();
      out[y * W + x] = buf[rank];
    }
  }
  return out;
}

export function sobel(frame, H, W, dtype) {
  const img = asFloat(frame, dtype), edge = [-1, 0, 1], smooth = [0.25, 0.5, 0.25];  // correlation form of [1, 0, -1]
  const gy = correlate1d(correlate1d(img, H, W, edge, 0, reflectIdx), H, W, smooth, 1, reflectIdx);
  const gx = correlate1d(correlate1d(img, H, W, smooth, 0, reflectIdx), H, W, edge, 1, reflectIdx);
  const out = new Float64Array(img.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt((gy[i] * gy[i] + gx[i] * gx[i]) / 2);
  return out;
}

export function laplace(frame, H, W, dtype) {
  const img = asFloat(frame, dtype), out = new Float64Array(img.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const at = (yy, xx) => img[reflectIdx(yy, H) * W + reflectIdx(xx, W)];
      out[y * W + x] = 4 * at(y, x) - at(y - 1, x) - at(y + 1, x) - at(y, x - 1) - at(y, x + 1);
    }
  }
  return out;
}

// ImageProcessor.apply_filter(image, filter_type, **params): returns { data, dtype } (sobel/laplacian give float64)
export function applyFilter(frame, H, W, dtype, type, p = {}) {
  if (type === 'gaussian') return { data: gaussian(frame, H, W, dtype, p.sigma), dtype };
  if (type === 'bilateral') return { data: gaussian(frame, H, W, dtype, p.sigma_spatial), dtype };
  if (type === 'median') return { data: median(frame, H, W, dtype, p.size), dtype };
  if (type === 'sobel') return { data: sobel(frame, H, W, dtype), dtype: 'float64' };
  if (type === 'laplacian') return { data: laplace(frame, H, W, dtype), dtype: 'float64' };
  throw new Error(`Unknown filter type: ${type}`);
}

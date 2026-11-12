// Node test: the JavaScript ports (filters.js, ops.js, stats.js) against the repository's Python code, using
// tests/fixtures/ops_cases.json (written by scripts/make_ops_fixture.py from ImageStack, ImageProcessor,
// DataAnalyzer and core/overlay_blend on a crop of the demo stacks). Run: node web/static/ops.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { applyFilter } from './filters.js';
import * as ops from './ops.js';
import { analyzeTrend, analyzePeaks } from './stats.js';

const here = dirname(fileURLToPath(import.meta.url));
const F = JSON.parse(readFileSync(join(here, '..', '..', 'tests', 'fixtures', 'ops_cases.json'), 'utf8'));
let failures = 0, checks = 0;
const report = [];

// compare arrays: |a - b| <= atol + rtol * |b|; integer outputs after truncation may differ by 1 count where the
// float result sits on an integer boundary (allowed for at most 0.5 % of pixels via `boundary`)
function same(name, got, want, { rtol = 1e-9, atol = 1e-9, boundary = 0 } = {}) {
  checks++;
  let bad = 0, worst = 0, off1 = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(Number(got[i]) - want[i]);
    if (d > atol + rtol * Math.abs(want[i])) {
      if (boundary && d <= 1) { off1++; continue; }
      bad++; worst = Math.max(worst, d);
    }
  }
  const okLen = got.length === want.length, ok = okLen && !bad && off1 <= boundary * want.length;
  if (!ok) { failures++; console.error(`FAIL ${name}: len ${got.length}/${want.length}, ${bad} differ (worst ${worst}), ${off1} off by one`); }
  report.push(`${ok ? 'ok  ' : 'FAIL'} ${name}`);
}
const close = (name, a, b, rtol = 1e-9) => same(name, [a], [b], { rtol });

const { T, H, W, dtype } = F.input, n = H * W;
const frames = Array.from({ length: T }, (_, t) => Uint16Array.from(F.input.data.slice(t * n, (t + 1) * n)));
const masks = Array.from({ length: T }, (_, t) => Uint8Array.from(F.input.mask.slice(t * n, (t + 1) * n)));
const f0 = frames[0];

for (const m of ['max', 'min', 'mean', 'median', 'std']) same(`projection ${m}`, ops.projection(frames, m, dtype).data, F.projection[m]);
same('get_normalized_frame', ops.normalizeFrame(f0), F.normalized);
same('apply_contrast 2-98 %', ops.applyContrast(frames, dtype, 2, 98)[0], F.apply_contrast.frame0, { boundary: 0.005 });
same('enhance_contrast rescale', ops.rescaleFrame(f0, 2, 98), F.enhance_rescale);
same('equalize_hist uint16', ops.equalizeHist(f0, dtype), F.equalize);
const ff = Float32Array.from(f0, v => v / 7);
same('equalize_hist float32', ops.equalizeHist(ff, 'float32'), F.equalize_float, { rtol: 1e-6, atol: 1e-7 });

const filt = (type, p, frame = f0, dt = dtype) => applyFilter(frame, H, W, dt, type, p).data;
same('gaussian sigma 1', filt('gaussian', { sigma: 1 }), F.filters.gaussian_1, { boundary: 0.005 });
same('gaussian sigma 2.5', filt('gaussian', { sigma: 2.5 }), F.filters['gaussian_2.5'], { boundary: 0.005 });
same('gaussian float32', filt('gaussian', { sigma: 1.5 }, Float32Array.from(f0, v => v / 3), 'float32'), F.filters.gaussian_f32, { rtol: 1e-6 });
same('bilateral (Gaussian of sigma_spatial)', filt('bilateral', { sigma_spatial: 2 }), F.filters.bilateral, { boundary: 0.005 });
same('median disk 1', filt('median', { size: 1 }), F.filters.median_1);
same('median disk 3', filt('median', { size: 3 }), F.filters.median_3);
same('sobel uint16', filt('sobel'), F.filters.sobel, { rtol: 1e-9, atol: 1e-12 });
same('sobel uint8 mask', filt('sobel', {}, masks[0], 'uint8'), F.filters.sobel_u8, { atol: 1e-12 });
same('laplacian', filt('laplacian'), F.filters.laplacian, { atol: 1e-12 });

close('otsu threshold uint16', ops.otsu(f0, dtype), F.otsu.threshold);
close('otsu threshold float32', ops.otsu(Float32Array.from(f0, v => v / 3), 'float32'), F.otsu.threshold_f32, 1e-6);
same('segment_image threshold', ops.segment(f0, dtype).mask, F.otsu.mask);

const keys = ['mean', 'std', 'min', 'max', 'sum', 'count'];
const sameStats = (name, got, want) => same(name, keys.map(k => got[k]), keys.map(k => want[k]));
sameStats('measure whole frame', ops.measure(f0), F.measure.whole);
sameStats('measure in mask', ops.measure(f0, masks[0]), F.measure.mask);
F.measure.region_defs.forEach((r, i) => sameStats(`measure ${r.type}`, ops.measure(f0, ops.regionMask(r, H, W)), F.measure.regions[i]));

const ts = ops.timeSeries(frames, masks[0]);
for (const k of ['mean', 'std', 'sum', 'max', 'min']) same(`time series in mask: ${k}`, ts[k], F.time_series_mask[k]);
const bs = ops.basicStats(frames);
for (const k of ['mean', 'median', 'std', 'min', 'max', 'snr']) same(`basic stats: ${k}`, bs[k], F.basic_stats[k]);

for (const [label, series, want] of [['bleaching cytoplasm', ts.mean, F.trend], ['puncta', F.peaks_input, F.trend_puncta]]) {
  const tr = analyzeTrend(series);
  for (const k of ['slope', 'intercept', 'r_value', 'std_err']) close(`trend ${label}: ${k}`, tr[k], want[k], 1e-8);
  close(`trend ${label}: p_value`, tr.p_value, want.p_value, 1e-6);
  checks++; if (tr.trend !== want.trend) { failures++; console.error(`FAIL trend label ${tr.trend} != ${want.trend}`); }
}
const pk = analyzePeaks(F.peaks_input);
same('find_peaks indices', pk.peak_indices, F.peaks.peak_indices);
same('find_peaks values', pk.peak_values, F.peaks.peak_values);

same('blend_overlay', ops.blendOverlay(f0, masks[0], F.blend.alpha, F.blend.color, dtype), F.blend.rgb);

console.log(report.join('\n'));
console.log(`ops: ${checks - failures}/${checks} checks agree with the Python code`);
process.exit(failures ? 1 : 0);

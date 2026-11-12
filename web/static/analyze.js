// Analyze page: the desktop's analysis on a channel, run in the browser with ports checked against Python:
// time series in a region (ImageProcessor.extract_time_series) with trend and peaks (DataAnalyzer), per-frame
// statistics (DataAnalyzer.calculate_basic_stats) and intensity in every region (ImageProcessor.measure_intensity).
import { route, app, esc, download } from './core.js';
import { S, ensureData, frameOf } from './session.js';
import { regionMask, measure } from './ops.js';
import { run } from './tasks.js';
import { lineChart, toCsv } from './charts.js';
import { fmt } from './render.js';

const g = v => (v == null || !Number.isFinite(v) ? '' : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(3) : fmt(v));

route('analyze', async () => {
  await ensureData();
  const regions = [['whole', 'Whole image'], ...(S.mask ? [['mask', `Mask "${S.mask.name}" of frame ${S.frame + 1} (as the desktop app)`]] : []),
    ...S.rois.map((r, i) => [`roi${i}`, `${r.label}: ${r.type}`])];
  app().innerHTML = `
  <h1>Analyze</h1>
  <p class="lead">Intensity over time inside a region, its linear trend and peaks, per-frame statistics, and region
  measurements, computed in your browser by ports of <code>core/image_processor.py</code> and
  <code>core/data_analyzer.py</code> (checked against the Python code). Draw ROIs on the Viewer page.</p>
  <form class="panel row" id="a-form">
    <div><label for="a-ch">Channel</label><select id="a-ch">${S.channels.map((c, i) => `<option value="${i}" ${i === S.selected ? 'selected' : ''}>${esc(c.name)}</option>`).join('')}</select></div>
    <div><label for="a-reg">Region</label><select id="a-reg">${regions.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join('')}</select></div>
    <div><button type="submit">Analyze</button></div>
  </form>
  <p id="a-msg" class="small mute" role="status"></p>
  <div id="a-out"></div>`;
  const root = app(), $ = s => root.querySelector(s);
  $('#a-form').addEventListener('submit', e => { e.preventDefault(); analyze(root); });
  if (!S.rois.length && S.mask) $('#a-reg').value = 'mask';
  await analyze(root);
});

function regionOf(key, c) {
  if (key === 'whole') return null;
  if (key === 'mask') return frameOf(S.mask, S.frame);
  return regionMask(S.rois[+key.slice(3)], c.H, c.W);
}

function area(m, c) {
  const px = m ? m.reduce((a, v) => a + (v > 0), 0) : c.H * c.W;
  return `${px.toLocaleString('en')} px${c.scaleKnown ? ` = ${(px * c.um_per_px ** 2).toFixed(1)} µm² (scale: ${c.scaleSource})` : ' (no pixel size in the file)'}`;
}

async function analyze(root) {
  const $ = s => root.querySelector(s), c = S.channels[+$('#a-ch').value], key = $('#a-reg').value;
  if (!c) { $('#a-out').innerHTML = '<p class="mute">Load a channel first.</p>'; return; }
  const m = regionOf(key, c), label = $('#a-reg').selectedOptions[0].textContent;
  $('#a-msg').textContent = 'Computing…';
  const [ts, bs] = await Promise.all([run('timeSeries', c.frames, m), run('basicStats', c.frames)]);
  const { trend, peaks } = await run('analyze', ts.mean);
  const unit = 'intensity (counts)', dt = c.frame_s;
  const regionRows = [['Whole image', null], ...(S.mask ? [[`Mask (frame ${S.frame + 1})`, frameOf(S.mask, S.frame)]] : []),
    ...S.rois.map(r => [r.label, regionMask(r, c.H, c.W)])].map(([n, mk]) => [n, measure(frameOf(c, S.frame), mk), mk]);
  $('#a-msg').textContent = `${c.name}: ${c.T} frames${dt ? `, ${dt} s per frame (from the file)` : ''}. Region area: ${area(m, c)}.`;
  $('#a-out').innerHTML = `
  <h2>Time series in ${esc(label)}</h2>
  <div class="panel">${lineChart({ series: [{ name: 'mean', values: ts.mean }], yLabel: `mean ${unit}`, trend, peaks: peaks.peak_indices })}
    <dl class="kv">
      <dt>trend</dt><dd><b>${esc(trend.trend)}</b> (p ≤ 0.05 counts as a trend, as DataAnalyzer._analyze_trend)</dd>
      <dt>slope</dt><dd>${g(trend.slope)} counts per frame${dt ? ` (${g(trend.slope / dt)} per s)` : ''} ± ${g(trend.std_err)}</dd>
      <dt>intercept, r, p</dt><dd>${g(trend.intercept)}, ${g(trend.r_value)}, ${g(trend.p_value)}</dd>
      <dt>peaks</dt><dd>${peaks.num_peaks} (frames ${peaks.peak_indices.map(i => i + 1).join(', ') || 'none'}; scipy find_peaks, height ≥ 0, distance 2)</dd>
    </dl>
    <button type="button" id="a-ts" class="alt">Download time series CSV</button></div>
  <h2>Per-frame statistics (whole image)</h2>
  <div class="panel">${lineChart({ series: [{ name: 'mean', values: bs.mean }, { name: 'median', values: bs.median }, { name: 'std', values: bs.std }], yLabel: unit })}
    ${lineChart({ series: [{ name: 'SNR = mean / std', values: bs.snr, color: '#ffd23f' }], yLabel: 'SNR', height: 160 })}
    <button type="button" id="a-bs" class="alt">Download statistics CSV</button></div>
  <h2>Regions at frame ${S.frame + 1}</h2>
  <div class="scroll"><table class="tbl"><thead><tr><th>region</th><th>mean</th><th>std</th><th>min</th><th>max</th><th>sum</th><th>pixels</th></tr></thead>
  <tbody>${regionRows.map(([n, s]) => `<tr><td>${esc(n)}</td>${s ? ['mean', 'std', 'min', 'max', 'sum', 'count'].map(k => `<td>${g(s[k])}</td>`).join('') : '<td colspan="6">empty</td>'}</tr>`).join('')}</tbody></table></div>
  <p class="small mute">Frame used for the regions table: the Viewer's current frame. Change it on the Viewer page.</p>`;
  const time = i => (dt ? [i + 1, +(i * dt).toFixed(6)] : [i + 1]);
  const head = dt ? ['frame', 'time_s'] : ['frame'];
  $('#a-ts').addEventListener('click', () => download(`time_series_${c.name}_${key}.csv`,
    toCsv([...head, 'mean', 'std', 'sum', 'max', 'min'], ts.mean.map((_, i) => [...time(i), ts.mean[i], ts.std[i], ts.sum[i], ts.max[i], ts.min[i]])), 'text/csv'));
  $('#a-bs').addEventListener('click', () => download(`frame_stats_${c.name}.csv`,
    toCsv([...head, 'mean', 'median', 'std', 'min', 'max', 'snr'], bs.mean.map((_, i) => [...time(i), bs.mean[i], bs.median[i], bs.std[i], bs.min[i], bs.max[i], bs.snr[i]])), 'text/csv'));
}

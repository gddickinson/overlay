// Small SVG line chart for per-frame series: lines, optional fitted trend line and peak markers.
import { esc } from './core.js';

const COLORS = ['#3dffc5', '#ff4fd8', '#ffb02e', '#7aa8ff', '#ffd23f', '#ff5c7a'];

export function lineChart({ series, xLabel = 'frame', yLabel = '', trend = null, peaks = [], height = 220 }) {
  const W = 640, H = height, L = 62, R = 12, T = 12, B = 36;
  const all = series.flatMap(s => s.values.filter(v => v != null && Number.isFinite(v)));
  if (!all.length) return '<p class="mute">No values.</p>';
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const n = Math.max(...series.map(s => s.values.length));
  const x = i => L + (n > 1 ? (i / (n - 1)) * (W - L - R) : (W - L - R) / 2), y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);
  const fmtT = v => (Math.abs(hi - lo) > 50 ? Math.round(v).toLocaleString('en') : v.toPrecision(3));
  const path = vals => vals.map((v, i) => (v == null || !Number.isFinite(v) ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`))
    .filter(Boolean).map((p, i) => (i ? 'L' : 'M') + p).join('');
  const xt = Array.from(new Set([0, Math.floor((n - 1) / 2), n - 1]));
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} per ${esc(xLabel)}">
    ${ticks.map(v => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtT(v)}</text>`).join('')}
    <line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
    ${xt.map(i => `<text x="${x(i)}" y="${H - B + 14}" text-anchor="middle">${i + 1}</text>`).join('')}
    <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>
    <text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">${esc(yLabel)}</text>
    ${series.map((s, k) => `<path d="${path(s.values)}" fill="none" stroke="${s.color || COLORS[k % COLORS.length]}" stroke-width="1.8"/>`).join('')}
    ${trend ? `<line class="trend" x1="${x(0)}" x2="${x(n - 1)}" y1="${y(trend.intercept)}" y2="${y(trend.intercept + trend.slope * (n - 1))}"/>` : ''}
    ${peaks.map(i => `<circle class="peak" cx="${x(i)}" cy="${y(series[0].values[i])}" r="4"/>`).join('')}
  </svg>
  <div class="legend">${series.map((s, k) => `<span class="key" style="--c:${s.color || COLORS[k % COLORS.length]}">${esc(s.name)}</span>`).join('')}
  ${trend ? '<span class="key trendkey">linear trend</span>' : ''}${peaks.length ? '<span class="key peakkey">peaks</span>' : ''}</div>`;
}

export const toCsv = (header, rows) => [header.join(','), ...rows.map(r => r.map(v => (v == null ? '' : String(v).includes(',') ? `"${v}"` : v)).join(','))].join('\n') + '\n';

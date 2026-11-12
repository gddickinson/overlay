// Ports of DataAnalyzer._analyze_trend (scipy.stats.linregress; p-value from Student's t, two-sided) and
// DataAnalyzer._find_peaks (scipy.signal.find_peaks(height=0, distance=2)). Checked by ops.test.mjs.

function logGamma(x) {  // Lanczos approximation (g = 7, n = 9)
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a, b, x) {  // continued fraction for the incomplete beta function (Numerical Recipes)
  const tiny = 1e-300;
  let c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return h;
}

// Regularized incomplete beta I_x(a, b)
export function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}

// Two-sided p-value of Student's t with df degrees of freedom
export const tTwoSided = (t, df) => betai(df / 2, 0.5, df / (df + t * t));

export function linregress(x, y) {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  sxx /= n; syy /= n; sxy /= n;
  let r = sxx === 0 || syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
  r = Math.max(-1, Math.min(1, r));
  const slope = sxy / sxx, intercept = my - slope * mx, df = n - 2;
  let p, se;
  if (n === 2) { p = 1; se = 0; } else {
    const t = r * Math.sqrt(df / ((1 - r + 1e-300) * (1 + r + 1e-300)));
    p = Math.abs(r) === 1 ? 0 : tTwoSided(t, df);
    se = Math.sqrt((1 - r * r) * syy / sxx / df);
  }
  return { slope, intercept, r_value: r, p_value: p, std_err: se };
}

export function analyzeTrend(data) {
  const clean = data.filter(v => v != null && Number.isFinite(v));
  if (clean.length < 2) return { slope: 0, intercept: 0, r_value: 0, p_value: 1, std_err: 0, trend: 'insufficient data' };
  const res = linregress(clean.map((_, i) => i), clean);
  res.trend = res.p_value <= 0.05 ? (res.slope > 0 ? 'increasing' : 'decreasing') : 'no significant trend';
  return res;
}

// scipy.signal.find_peaks: local maxima (plateaus give their middle sample), height >= minHeight, then peaks closer
// than `distance` to a higher peak are dropped.
export function findPeaks(x, { height = 0, distance = 2 } = {}) {
  let peaks = [];
  for (let i = 1; i < x.length - 1;) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < x.length - 1 && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) { peaks.push(Math.floor((i + ahead - 1) / 2)); i = ahead; continue; }
    }
    i++;
  }
  peaks = peaks.filter(p => x[p] >= height);
  const keep = peaks.map(() => true), order = peaks.map((_, i) => i).sort((a, b) => x[peaks[a]] - x[peaks[b]] || a - b);
  for (let k = order.length - 1; k >= 0; k--) {
    const j = order[k];
    if (!keep[j]) continue;
    for (let i = j - 1; i >= 0 && peaks[j] - peaks[i] < distance; i--) keep[i] = false;
    for (let i = j + 1; i < peaks.length && peaks[i] - peaks[j] < distance; i++) keep[i] = false;
  }
  return peaks.filter((_, i) => keep[i]);
}

export function analyzePeaks(data) {
  const idx = data.map((v, i) => [v, i]).filter(([v]) => v != null && Number.isFinite(v));
  if (idx.length < 3) return { peak_indices: [], peak_values: [], num_peaks: 0 };
  const p = findPeaks(idx.map(([v]) => v));
  return { peak_indices: p.map(i => idx[i][1]), peak_values: p.map(i => idx[i][0]), num_peaks: p.length };
}

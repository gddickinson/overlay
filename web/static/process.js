// Process page: the repository's ImageProcessor / ImageStack operations on a channel, in the browser (JS ports,
// checked against Python) or, in the local app, in Python. Results become new channels or the mask; the input is
// never changed (the desktop app overwrites the frame in place).
import { route, app, api, esc, isStatic, needsServer } from './core.js';
import { S, ensureData, addChannel, setMask, frameOf } from './session.js';
import { paint, fmt } from './render.js';
import { run } from './tasks.js';
import { toB64, fromB64, stackB64 } from './b64.js';
import { percentile, minMax } from './ops.js';

let last = null;

route('process', async () => {
  await ensureData();
  const { operations } = await api('/ops');
  const groups = [...new Set(operations.map(o => o.group))];
  app().innerHTML = `
  <h1>Process</h1>
  <p class="lead">Filters, contrast, segmentation, feature detection and registration from <code>core/image_processor.py</code>.
  Operations marked <span class="badge js">browser</span> run here, ported to JavaScript and checked against the Python
  code; <span class="badge py">Python</span> ones need the local app.</p>
  <div class="side">
    <form class="panel" id="p-form">
      <label for="p-ch">Channel</label><select id="p-ch">${S.channels.map((c, i) => `<option value="${i}">${esc(c.name)}</option>`).join('')}</select>
      <label for="p-op">Operation</label>
      <select id="p-op">${groups.map(g => `<optgroup label="${g}">${operations.filter(o => o.group === g).map(o =>
        `<option value="${o.id}">${esc(o.label)}${o.where === 'python' ? ' [Python]' : ''}</option>`).join('')}</optgroup>`).join('')}</select>
      <div id="p-params"></div>
      <label for="p-scope">Apply to</label>
      <select id="p-scope"><option value="frame">current frame (as the desktop app)</option><option value="all">all frames</option></select>
      <label for="p-where">Run in</label>
      <select id="p-where"><option value="js">this browser (JS port)</option><option value="python" ${isStatic() ? 'disabled' : ''}>Python (local app)</option></select>
      <label for="p-frame">Frame <b id="p-fl"></b></label><input type="range" id="p-frame" min="0" step="1">
      <button type="submit" id="p-run">Run</button>
      <p id="p-msg" class="small" role="status"></p>
    </form>
    <div>
      <div class="two"><figure><div class="stage"><canvas id="p-before" role="img" aria-label="Input frame"></canvas></div><figcaption class="small mute">Input</figcaption></figure>
      <figure><div class="stage"><canvas id="p-after" role="img" aria-label="Result"></canvas></div><figcaption class="small mute" id="p-cap">Result</figcaption></figure></div>
      <div id="p-out"></div>
    </div>
  </div>`;
  const root = app(), $ = s => root.querySelector(s), op = () => operations.find(o => o.id === $('#p-op').value);
  const ch = () => S.channels[+$('#p-ch').value];
  const say = (m, bad) => { $('#p-msg').textContent = m; $('#p-msg').className = 'small ' + (bad ? 'err' : 'mute'); };
  const fr = $('#p-frame');
  fr.max = Math.max(0, (ch()?.T || 1) - 1); fr.value = Math.min(S.frame, +fr.max);
  const auto = d => { const lo = percentile(d, 1), hi = percentile(d, 99); return hi > lo ? [lo, hi] : minMax(d); };
  const show = (canvas, data, kind = 'image', points = []) => {
    const c = ch(), [lo, hi] = kind === 'mask' ? [0, 1] : auto(data);
    paint(canvas, { channels: [{ min: lo, max: hi || lo + 1, lut: 'Grays' }], planesFor: () => data, W: c.W, H: c.H, points });
  };
  const drawInput = () => { $('#p-fl').textContent = `${+fr.value + 1} / ${ch().T}`; show($('#p-before'), frameOf(ch(), +fr.value)); };
  const params = () => Object.fromEntries(op().params.map(p => [p.key, +$(`#pp-${p.key}`).value]));
  const setOp = () => {
    const o = op();
    $('#p-params').innerHTML = o.params.map(p => `<label for="pp-${p.key}">${esc(p.label)} <b id="pv-${p.key}">${p.default}</b></label>
      <input type="range" id="pp-${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">`).join('')
      + (o.where === 'python' && isStatic() ? needsServer(o.label) : '') + (o.params.length ? '<p class="small mute">Initial values: the defaults in core/processing_defaults.py.</p>' : '');
    o.params.forEach(p => $(`#pp-${p.key}`).addEventListener('input', e => { $(`#pv-${p.key}`).textContent = e.target.value; }));
    $('#p-where').value = o.where === 'python' ? 'python' : $('#p-where').value;
    $('#p-where').querySelector('[value=js]').disabled = o.where === 'python';
    $('#p-scope').value = o.stack_only ? 'all' : o.frame_only ? 'frame' : $('#p-scope').value;
    $('#p-scope').disabled = !!(o.stack_only || o.frame_only);
    $('#p-run').disabled = o.where === 'python' && isStatic();
  };
  $('#p-op').addEventListener('change', setOp);
  $('#p-ch').addEventListener('change', () => { fr.max = ch().T - 1; drawInput(); });
  fr.addEventListener('input', drawInput);
  setOp(); drawInput();

  $('#p-form').addEventListener('submit', async e => {
    e.preventDefault();
    const o = op(), c = ch(), p = params(), all = $('#p-scope').value === 'all', t = +fr.value;
    const frames = all ? c.frames : [frameOf(c, t)];
    $('#p-run').disabled = true;
    const t0 = performance.now();
    try {
      say('Running…');
      last = $('#p-where').value === 'python' ? await viaPython(o, p, frames, c, t, all, say) : await viaJs(o, p, frames, c);
      last.label = `${c.name} · ${o.id}${o.params.length ? ' ' + o.params.map(q => `${q.key}=${p[q.key]}`).join(' ') : ''}${all ? '' : ` · frame ${t + 1}`}`;
      say(`Done in ${((performance.now() - t0) / 1000).toFixed(2)} s (${$('#p-where').value === 'python' ? 'Python' : 'browser'}).`);
      showResult(root, show, c, t, all);
    } catch (err) { say(err.message, true); }
    $('#p-run').disabled = false;
  });
  $('#p-form').requestSubmit();  // show a first result straight away (the default operation, in the browser)
});

async function viaJs(o, p, frames, c) {
  const { H, W, dtype } = c;
  if (['gaussian', 'median', 'bilateral', 'sobel', 'laplacian'].includes(o.id)) {
    const r = await run('filterStack', frames, H, W, dtype, o.id, p);
    return { kind: 'image', frames: r.map(x => x.data), dtype: r[0].dtype };
  }
  if (o.id === 'rescale') return { kind: 'image', frames: await run('rescaleStack', frames, p.p_low, p.p_high), dtype: 'float64' };
  if (o.id === 'stack_contrast') return { kind: 'image', frames: await run('applyContrast', frames, dtype, p.p_low, p.p_high), dtype };
  if (o.id === 'equalize') return { kind: 'image', frames: await run('equalizeStack', frames, dtype), dtype: dtype === 'float32' ? 'float32' : 'float64' };
  if (o.id === 'threshold') {
    const r = await run('segmentStack', frames, dtype);
    return { kind: 'mask', frames: r.map(x => x.mask), dtype: 'uint8', note: `Otsu threshold ${r.map(x => fmt(x.threshold)).slice(0, 6).join(', ')}${r.length > 6 ? ', …' : ''}` };
  }
  throw new Error(`${o.label} runs in Python only.`);
}

async function viaPython(o, p, frames, c, t, all, say) {
  if (o.id === 'register') {
    const r = await api('/register', { body: { frames: stackB64(c.frames, c.H, c.W, c.dtype), reference: t, params: p } });
    const flat = fromB64(r.frames), n = c.H * c.W;
    return { kind: 'image', frames: Array.from({ length: c.T }, (_, i) => flat.slice(i * n, (i + 1) * n)), dtype: r.frames.dtype,
      note: `Registered to frame ${t + 1}. Shifts (dy, dx) px: ${r.shifts.slice(0, 8).map(s => s ? s.map(v => v.toFixed(2)).join(', ') : 'n/a').join(' | ')}${r.shifts.length > 8 ? ' …' : ''}` };
  }
  if (o.id === 'stack_contrast') throw new Error('This one runs in the browser.');
  const out = [];
  let kind = 'image', dtype = c.dtype, points = [];
  for (let i = 0; i < frames.length; i++) {
    if (frames.length > 1) say(`Python: frame ${i + 1} / ${frames.length}…`);
    const r = await api('/process', { body: { op: o.id, params: p, frame: toB64(frames[i], [c.H, c.W], c.dtype) } });
    kind = r.kind;
    if (r.kind === 'points') { points = r.points; out.push(frames[i]); } else { out.push(fromB64(r.image)); dtype = r.image.dtype; }
  }
  return { kind, frames: out, dtype, points, note: kind === 'points' ? `${points.length} features on frame ${t + 1} (y, x, size)` : '' };
}

function showResult(root, show, c, t, all) {
  const r = last, $ = s => root.querySelector(s), f = r.frames[all ? Math.min(t, r.frames.length - 1) : 0];
  show($('#p-after'), f, r.kind === 'mask' ? 'mask' : 'image', r.points || []);
  $('#p-cap').textContent = `${r.label} (${r.dtype}${r.frames.length > 1 ? `, ${r.frames.length} frames; frame ${t + 1} shown` : ''})`;
  const btn = r.kind === 'mask' ? '<button type="button" id="p-use">Use as mask</button>' : r.kind === 'image' ? '<button type="button" id="p-add">Add as new channel</button>' : '';
  const rows = (r.points || []).slice(0, 50).map(q => `<tr>${q.map(v => `<td>${fmt(v)}</td>`).join('')}</tr>`).join('');
  $('#p-out').innerHTML = `${r.note ? `<p class="small">${esc(r.note)}</p>` : ''}${btn}
    ${rows ? `<div class="scroll"><table class="tbl"><thead><tr><th>y</th><th>x</th><th>${r.points[0].length > 2 ? 'sigma' : ''}</th></tr></thead><tbody>${rows}</tbody></table></div>` : ''}
    <p class="small mute">Shown with automatic 1-99 % levels. Added results appear in the Viewer and Analyze pages.</p>`;
  const add = $('#p-add'), use = $('#p-use');
  if (add) add.addEventListener('click', () => { addChannel({ name: r.label, frames: r.frames, T: r.frames.length, H: c.H, W: c.W, dtype: r.dtype,
    meta: { um_per_px: c.um_per_px, scaleKnown: c.scaleKnown, scaleSource: c.scaleSource, frame_s: c.frame_s, layout: 'computed here' } });
    add.disabled = true; add.textContent = 'Added'; });
  if (use) use.addEventListener('click', () => { setMask({ name: r.label, frames: r.frames, T: r.frames.length, H: c.H, W: c.W }); use.disabled = true; use.textContent = 'Mask set'; });
}

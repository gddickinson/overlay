// Viewer side panel: file loading, the channel list (visibility, LUT, min/max, auto levels), mask settings, view mode
// (frame or a Z-projection, as ImageStack.create_z_projection) and ROIs. Calls redraw() after every change.
import { esc } from './core.js';
import { S, loadTiff, loadDemo, removeChannel, clearAll, autoLevels, changed } from './session.js';
import { fmt } from './render.js';

const PROJ = [['frame', 'Current frame'], ['max', 'Max projection'], ['min', 'Min projection'], ['mean', 'Mean projection'],
  ['median', 'Median projection'], ['std', 'Std projection']];

export function panelHTML() {
  return `
  <div class="panel vp">
    <h3>Open TIFF stacks</h3>
    <label for="v-files">TIFF files (.tif, .tiff, .ome.tif)</label>
    <input type="file" id="v-files" accept=".tif,.tiff,image/tiff" multiple>
    <label for="v-role">Load as</label>
    <select id="v-role"><option value="channel">channel(s)</option><option value="mask">mask (values > 0 are inside)</option></select>
    <p class="small mute">Read in your browser; nothing is uploaded. Or drop files on the image. Your first file replaces the demo.</p>
    <button type="button" id="v-demo" class="alt">Demo stacks</button><button type="button" id="v-clear" class="alt">Clear</button>
    <p id="v-msg" class="small" role="status"></p>
  </div>
  <div class="panel vp"><h3>Channels</h3><div id="v-chans"></div></div>
  <div class="panel vp" id="v-maskbox"></div>
  <div class="panel vp">
    <h3>View</h3>
    <label for="v-view">Show</label>
    <select id="v-view">${PROJ.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>
    <label for="v-tool">Draw ROI on the image</label>
    <select id="v-tool"><option value="">off</option><option value="rectangle">rectangle</option><option value="circle">circle</option></select>
    <div id="v-rois" class="small"></div>
  </div>`;
}

function channelHTML(ch, i) {
  const lutOpts = Object.keys(S.luts).map(n => `<option ${n === ch.lut ? 'selected' : ''}>${esc(n)}</option>`).join('');
  const step = ch.dtype.startsWith('float') ? (ch.hi - ch.lo) / 1000 : 1;
  const sel = i === S.selected;
  return `<div class="chan ${sel ? 'sel' : ''}" data-id="${ch.id}">
    <div class="chead"><label class="inline"><input type="checkbox" data-k="visible" ${ch.visible ? 'checked' : ''}> <b>${esc(ch.name)}</b></label>
      <button type="button" class="x" data-k="remove" aria-label="Remove ${esc(ch.name)}">×</button></div>
    <small class="mute">${ch.T} frames · ${ch.W} × ${ch.H} · ${ch.dtype}${ch.scaleKnown ? ` · ${(+ch.um_per_px).toPrecision(4)} µm/px` : ''}</small>
    <label for="lut-${ch.id}">LUT</label><select id="lut-${ch.id}" data-k="lut">${lutOpts}</select>
    <label for="min-${ch.id}">Min <b>${fmt(ch.min)}</b></label>
    <input type="range" id="min-${ch.id}" data-k="min" min="${ch.lo}" max="${ch.hi}" step="${step}" value="${ch.min}">
    <label for="max-${ch.id}">Max <b>${fmt(ch.max)}</b></label>
    <input type="range" id="max-${ch.id}" data-k="max" min="${ch.lo}" max="${ch.hi}" step="${step}" value="${ch.max}">
    <button type="button" class="alt sm" data-k="auto">Auto (1-99 %)</button><button type="button" class="alt sm" data-k="full">Full range</button>
    <button type="button" class="alt sm" data-k="select" ${sel ? 'disabled' : ''}>${sel ? 'histogram shown' : 'show histogram'}</button>
  </div>`;
}

function maskHTML() {
  if (!S.mask) return '<h3>Mask</h3><p class="small mute">No mask loaded. Load a TIFF "as mask", or make one on the Process page (Otsu threshold).</p>';
  const colors = Object.entries(S.maskColors).map(([n, c]) => `<option value="${c.join(',')}" ${c.join() === S.maskColor.join() ? 'selected' : ''}>${n}</option>`).join('');
  return `<h3>Mask</h3>
    <label class="inline"><input type="checkbox" id="m-vis" ${S.mask.visible ? 'checked' : ''}> show <b>${esc(S.mask.name)}</b></label>
    <small class="mute">${S.mask.T} frames · ${S.mask.W} × ${S.mask.H}</small>
    <label for="m-col">Mask colour</label><select id="m-col">${colors}</select>
    <label for="m-alpha">Opacity <b id="m-av">${S.maskAlpha.toFixed(2)}</b></label>
    <input type="range" id="m-alpha" min="0" max="1" step="0.01" value="${S.maskAlpha}">
    <button type="button" class="alt sm" id="m-del">Remove mask</button>`;
}

export function renderRois(root) {
  const host = root.querySelector('#v-rois');
  host.innerHTML = S.rois.length ? S.rois.map((r, i) => `<div class="roi">${esc(r.label)}: ${r.type} ${r.points.map(v => Math.round(v * 10) / 10).join(', ')}
    <button type="button" class="x" data-roi="${i}" aria-label="Delete ${esc(r.label)}">×</button></div>`).join('')
    : '<p class="mute">Drag on the image to add one; ROIs are used on the Analyze page.</p>';
  host.querySelectorAll('[data-roi]').forEach(b => b.addEventListener('click', () => { S.rois.splice(+b.dataset.roi, 1); changed('rois'); }));
}

export function renderPanel(root, redraw) {
  const host = root.querySelector('#v-chans');
  host.innerHTML = S.channels.length ? S.channels.map(channelHTML).join('') : '<p class="small mute">No channels loaded.</p>';
  host.querySelectorAll('.chan').forEach(div => {
    const ch = S.channels.find(c => c.id === +div.dataset.id);
    div.querySelectorAll('[data-k]').forEach(el => {
      const k = el.dataset.k, ev = el.type === 'range' ? 'input' : el.tagName === 'BUTTON' ? 'click' : 'change';
      el.addEventListener(ev, () => {
        if (k === 'visible') ch.visible = el.checked;
        else if (k === 'lut') ch.lut = el.value;
        else if (k === 'min' || k === 'max') {
          ch[k] = +el.value;
          if (ch.min >= ch.max) ch[k === 'min' ? 'max' : 'min'] = ch[k] + (k === 'min' ? 1 : -1) * (ch.hi - ch.lo) / 1000;
          el.previousElementSibling.querySelector('b').textContent = fmt(ch[k]);
          redraw(); return;
        } else if (k === 'auto') autoLevels(ch, redraw.shown(ch));
        else if (k === 'full') { ch.min = ch.lo; ch.max = ch.hi; }
        else if (k === 'select') S.selected = S.channels.indexOf(ch);
        else if (k === 'remove') { removeChannel(ch.id); return; }
        renderPanel(root, redraw); redraw();
      });
    });
  });
  const mb = root.querySelector('#v-maskbox');
  mb.innerHTML = maskHTML();
  if (S.mask) {
    mb.querySelector('#m-vis').addEventListener('change', e => { S.mask.visible = e.target.checked; redraw(); });
    mb.querySelector('#m-col').addEventListener('change', e => { S.maskColor = e.target.value.split(',').map(Number); redraw(); });
    mb.querySelector('#m-alpha').addEventListener('input', e => { S.maskAlpha = +e.target.value; mb.querySelector('#m-av').textContent = S.maskAlpha.toFixed(2); redraw(); });
    mb.querySelector('#m-del').addEventListener('click', () => { S.mask = null; changed('mask'); });
  }
  renderRois(root);
}

export function bindLoaders(root, say) {
  const files = async list => {
    const role = root.querySelector('#v-role').value;
    for (const f of list) {
      try {
        say(`Reading ${f.name}…`);
        const { st, added } = await loadTiff(await f.arrayBuffer(), f.name, role);
        say(`${f.name}: ${st.T * st.Z} frames × ${st.C} channel(s), ${st.W} × ${st.H} ${st.dtype} (${st.layout}); scale: ${st.scaleSource}. Added ${added.join(', ')}.`);
      } catch (e) { say(`${f.name}: ${e.message}`, true); }
    }
  };
  root.querySelector('#v-files').addEventListener('change', e => files([...e.target.files]));
  root.querySelector('#v-demo').addEventListener('click', async () => { say('Loading the demo stacks…'); await loadDemo(); say('Demo loaded: synthetic data from scripts/make_demo_data.py.'); });
  root.querySelector('#v-clear').addEventListener('click', () => { clearAll(); say('Cleared.'); });
  return files;
}

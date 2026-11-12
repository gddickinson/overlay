// Home page: hero with the demo composite, counts from the API, entry cards.
import { route, api, app, esc } from './core.js';
import { S, ensureData, frameOf } from './session.js';
import { paint } from './render.js';

const DOORS = [
  ['#/viewer', 'Viewer', 'Open TIFF stacks (drop files or pick them), overlay channels with LUTs and contrast, add a mask, play frames, project, read pixel values, draw ROIs, save a PNG.'],
  ['#/process', 'Process', 'Gaussian, median, Sobel, Laplacian, contrast stretch, histogram equalisation and Otsu masks in the browser; Frangi, CLAHE, local threshold, blobs, corners and registration in the local app.'],
  ['#/analyze', 'Analyze', 'Intensity over time in the mask or an ROI with its linear trend and peaks, per-frame statistics, region measurements, CSV export.'],
  ['#/about', 'About', 'What the desktop app does, how the browser version works, which TIFFs it reads, and how the ports are checked against the Python code.'],
];

route('home', async () => {
  const [info] = await Promise.all([api('/info'), ensureData()]);
  const c = info.counts;
  app().innerHTML = `
  <section class="hero two">
    <div>
      <p class="kicker">Microscopy TIFF stacks · overlay · analysis</p>
      <h1 class="glow">TIFF Stack Viewer</h1>
      <p class="lead">${esc(info.description)} A browser companion to the PyQt6 desktop app in this repository, built
      for fluorescence stacks overlaid with a binary mask.</p>
      <a class="btn" href="#/viewer">Open the viewer</a><a class="btn alt" href="#/analyze">Analyze the demo</a>
    </div>
    <figure class="heroimg"><div class="stage"><canvas id="h-img" role="img" aria-label="Demo composite: two channels and a mask outline"></canvas></div>
      <figcaption class="small mute" id="h-cap"></figcaption></figure>
  </section>
  <div class="tiles">
    <div class="tile"><b>${c.ops_js}</b><span>operations in the browser</span></div>
    <div class="tile"><b>${c.ops_python}</b><span>more in the local app</span></div>
    <div class="tile"><b>${c.luts}</b><span>LUTs</span></div>
    <div class="tile"><b>${c.demos}</b><span>demo stacks</span></div>
  </div>
  <div class="grid wide">${DOORS.map(([href, t, d]) => `<a class="card" href="${href}"><b>${t}</b><span>${d}</span></a>`).join('')}</div>`;
  const d = S.channels[0];
  if (!d) return;
  const t = Math.min(12, d.T - 1);
  paint(document.getElementById('h-img'), { channels: S.channels.filter(ch => ch.visible), planesFor: ch => frameOf(ch, t), W: d.W, H: d.H,
    mask: S.mask ? frameOf(S.mask, t) : null, alpha: S.maskAlpha * 0.5, color: S.maskColor, scale: S.channels.find(ch => ch.scaleKnown) });
  document.getElementById('h-cap').textContent = `Frame ${t + 1} of the synthetic demo: ${S.channels.map(ch => `${ch.name} (${ch.lut})`).join(', ')}${S.mask ? `, mask in ${Object.keys(S.maskColors).find(k => S.maskColors[k].join() === S.maskColor.join()) || 'colour'}` : ''}.`;
});

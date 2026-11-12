// Entry point: lazy page modules, the mode label, a faint drifting-puncta background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'viewer.js': ['viewer'], 'process.js': ['process'], 'analyze.js': ['analyze'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: TIFF reading, display, ported processing and analysis run in your browser; Python-only operations need the local app'
  : 'local app: the browser does the display; Python operations run on this computer';

// Background: a few soft fluorescent spots slowly blinking. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cols = ['61,255,197', '255,79,216', '255,176,46'];
  const spots = Array.from({ length: 18 }, (_, i) => ({ x: Math.random(), y: Math.random(), r: 20 + Math.random() * 50,
    c: cols[i % 3], ph: Math.random() * 6.28, sp: 0.0004 + Math.random() * 0.0008 }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const s of spots) {
      const a = 0.25 + 0.25 * Math.sin(s.ph + t * s.sp), x = s.x * canvas.width, y = s.y * canvas.height;
      const grad = g.createRadialGradient(x, y, 0, x, y, s.r);
      grad.addColorStop(0, `rgba(${s.c},${a})`); grad.addColorStop(1, `rgba(${s.c},0)`);
      g.fillStyle = grad; g.fillRect(x - s.r, y - s.r, 2 * s.r, 2 * s.r);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();

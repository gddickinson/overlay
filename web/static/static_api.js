// Static-site data layer: answers api() GETs from the JSON exported by scripts/build_static.py. Viewing, the
// ported processing and the analysis run in the browser and never call the API; the Python-only operations
// (POST /process, /register) need the local app.
export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body || clean === 'process' || clean === 'register') throw new Error('Python processing ' + NEEDS_SERVER + '.');
  return file(clean);
}

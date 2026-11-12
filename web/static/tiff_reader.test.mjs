// Node test: the browser TIFF reader against tifffile. Every file in tests/fixtures/tiff/ (written by
// scripts/make_tiff_fixtures.py) must give tifffile's shape, dtype, scale and pixel checksums
// (tests/fixtures/tiff_expected.json); the demo stacks must read as the desktop app reads them.
// Run: node web/static/tiff_reader.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readTiff, TiffError } from './tiff_reader.js';
import { lzwDecode, packbitsDecode } from './tiff_decode.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, '..', '..', 'tests', 'fixtures');
const expected = JSON.parse(readFileSync(join(fixtures, 'tiff_expected.json'), 'utf8'));
let failures = 0, checks = 0;
const ok = (cond, msg) => { checks++; if (!cond) { failures++; console.error('FAIL', msg); } };
const close = (a, b, rel = 1e-9) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));
const load = p => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

function checksums(st) {
  let sum = 0, sumsq = 0, weighted = 0, i = 0, min = Infinity, max = -Infinity;
  const first = [];
  for (let t = 0; t < st.T; t++) for (let z = 0; z < st.Z; z++) for (let c = 0; c < st.C; c++) {
    const p = st.plane(t, z, c);
    for (let k = 0; k < p.length; k++, i++) {
      const v = Number(p[k]);
      sum += v; sumsq += v * v; weighted += v * ((i % 97) + 1);
      if (v < min) min = v; if (v > max) max = v;
      if (first.length < 8) first.push(v);
    }
  }
  return { sum, sumsq, weighted, min, max, first };
}

for (const [name, e] of Object.entries(expected)) {
  let st;
  try { st = await readTiff(load(join(fixtures, 'tiff', name))); } catch (err) { ok(false, `${name}: ${err.message}`); continue; }
  for (const k of ['T', 'Z', 'C', 'H', 'W', 'dtype']) ok(st[k === 'H' ? 'H' : k] === e[k], `${name}: ${k} ${st[k]} != ${e[k]}`);
  const cs = checksums(st);
  for (const k of ['sum', 'sumsq', 'weighted', 'min', 'max']) ok(close(cs[k], e[k]), `${name}: ${k} ${cs[k]} != ${e[k]}`);
  ok(cs.first.every((v, i) => close(v, e.first[i])), `${name}: first values ${cs.first} != ${e.first}`);
  if (e.um_per_px) ok(st.scaleKnown && close(st.um_per_px, e.um_per_px, 1e-6), `${name}: um_per_px ${st.um_per_px} != ${e.um_per_px}`);
  else ok(!st.scaleKnown, `${name}: unexpected scale ${st.um_per_px} (${st.scaleSource})`);
  if (e.z_um) ok(close(st.z_um, e.z_um, 1e-6), `${name}: z spacing ${st.z_um}`);
  if (e.frame_s) ok(close(st.frame_s, e.frame_s, 1e-6), `${name}: frame interval ${st.frame_s}`);
}

// demo stacks: (T, Y, X) as the desktop's ImageStack.load_tiff reads them
const demo = join(here, 'demo');
for (const [f, dtype] of [['demo_cytoplasm.tif', 'uint16'], ['demo_puncta.tif', 'uint16'], ['demo_mask.tif', 'uint8']]) {
  const st = await readTiff(load(join(demo, f)));
  ok(st.T === 24 && st.Z === 1 && st.C === 1 && st.W === 128 && st.H === 128 && st.dtype === dtype, `${f}: layout ${st.T}x${st.C}x${st.H}x${st.W} ${st.dtype}`);
  if (f !== 'demo_mask.tif') ok(st.scaleKnown && close(st.um_per_px, 0.2, 1e-6), `${f}: pixel size ${st.um_per_px}`);
}

// codecs on hand-made inputs; refusals with a clear message
ok(packbitsDecode(Uint8Array.from([0xfe, 7, 2, 1, 2, 3]), 6).join() === '7,7,7,1,2,3', 'packbits');
ok(lzwDecode(Uint8Array.from([0x80, 0x00, 0x00, 0x00]), 0).length === 0, 'lzw empty');
for (const [bytes, msg] of [[[1, 2, 3, 4, 5, 6, 7, 8], 'Not a TIFF'], [[0x49, 0x49, 42, 0, 0, 0, 0, 0], 'No images']]) {
  let err = null;
  try { await readTiff(Uint8Array.from(bytes).buffer); } catch (e) { err = e; }
  ok(err instanceof TiffError && err.message.includes(msg), `refusal "${msg}": ${err && err.message}`);
}

console.log(`tiff_reader: ${checks - failures}/${checks} checks passed over ${Object.keys(expected).length} fixtures + 3 demo stacks`);
process.exit(failures ? 1 : 0);

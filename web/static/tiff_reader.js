// Pure-JS TIFF reader for microscopy stacks (no library). Classic TIFF and BigTIFF, either byte order; 8/16/32-bit
// integer and 32/64-bit float samples; strips or tiles; uncompressed, LZW, Deflate or PackBits with predictor 2 or 3;
// one or more samples per pixel (chunky or planar). Pixel data stay in their own type (Uint16Array, ...).
//
// Layout: pages (and samples) are arranged into T x Z x C planes from, in order, an ImageJ description (frames,
// slices, channels; channels fastest), OME-XML (SizeT/Z/C and DimensionOrder) or tifffile's shaped description
// ({"shape": ..., "axes": ...}); otherwise every page is a frame. Axes other than Z and C (I, Q, ...) count as time.
// Scale: ImageJ unit + XResolution, OME PhysicalSizeX, or a resolution in cm; otherwise unknown (scaleKnown false),
// and the result says which source was used. Tested against tifffile: tiff_reader.test.mjs.
import { TiffError, decodeChunk } from './tiff_decode.js';

export { TiffError };
const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8 };
const UNIT_UM = { micron: 1, microns: 1, um: 1, 'µm': 1, 'μm': 1, '\\u00B5m': 1, '\\u00b5m': 1, nm: 1e-3, mm: 1e3, cm: 1e4, m: 1e6 };
const ARRAYS = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array,
  int32: Int32Array, float32: Float32Array, float64: Float64Array };

function readIfd(dv, off, le, big) {
  const n = big ? Number(dv.getBigUint64(off, le)) : dv.getUint16(off, le);
  const head = big ? 8 : 2, size = big ? 20 : 12, tags = {};
  for (let i = 0; i < n; i++) {
    const e = off + head + i * size, tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le);
    const count = big ? Number(dv.getBigUint64(e + 4, le)) : dv.getUint32(e + 4, le);
    const bytes = (TYPE_SIZE[type] || 1) * count, inline = bytes <= (big ? 8 : 4);
    const at = inline ? e + (big ? 12 : 8) : (big ? Number(dv.getBigUint64(e + 12, le)) : dv.getUint32(e + 8, le));
    if (type === 2) {
      tags[tag] = new TextDecoder('utf-8').decode(new Uint8Array(dv.buffer, at, count)).replace(/\0+$/, '');
      continue;
    }
    if (count > 1e7) continue;
    const vals = new Array(count);
    for (let k = 0; k < count; k++) {
      if (type === 3) vals[k] = dv.getUint16(at + 2 * k, le);
      else if (type === 8) vals[k] = dv.getInt16(at + 2 * k, le);
      else if (type === 4 || type === 13) vals[k] = dv.getUint32(at + 4 * k, le);
      else if (type === 9) vals[k] = dv.getInt32(at + 4 * k, le);
      else if (type === 5) vals[k] = dv.getUint32(at + 8 * k, le) / dv.getUint32(at + 8 * k + 4, le);
      else if (type === 10) vals[k] = dv.getInt32(at + 8 * k, le) / dv.getInt32(at + 8 * k + 4, le);
      else if (type === 1 || type === 7) vals[k] = dv.getUint8(at + k);
      else if (type === 6) vals[k] = dv.getInt8(at + k);
      else if (type === 11) vals[k] = dv.getFloat32(at + 4 * k, le);
      else if (type === 12) vals[k] = dv.getFloat64(at + 8 * k, le);
      else if (type === 16 || type === 18) vals[k] = Number(dv.getBigUint64(at + 8 * k, le));
      else if (type === 17) vals[k] = Number(dv.getBigInt64(at + 8 * k, le));
      else vals[k] = 0;
    }
    tags[tag] = vals;
  }
  const nextAt = off + head + n * size;
  return { tags, next: big ? Number(dv.getBigUint64(nextAt, le)) : dv.getUint32(nextAt, le) };
}

export function sampleType(bits, fmt) {
  const t = fmt === 3 ? `float${bits}` : `${fmt === 2 ? 'int' : 'uint'}${bits}`;
  if (!ARRAYS[t]) throw new TiffError(`${bits}-bit ${fmt === 3 ? 'float' : 'integer'} samples are not supported.`);
  return t;
}

// Decode one page into `spp` planes of its sample type.
async function readPage(buf, dv, t, le) {
  const W = t[256][0], H = t[257][0], spp = (t[277] || [1])[0], bits = (t[258] || [1])[0];
  const dtype = sampleType(bits, (t[339] || [1])[0]), bps = bits / 8, planar = (t[284] || [1])[0] === 2;
  const tiled = !!t[322], cw = tiled ? t[322][0] : W, ch = tiled ? t[323][0] : Math.min(H, (t[278] || [H])[0]);
  const offs = tiled ? t[324] : t[273], counts = tiled ? t[325] : t[279];
  if (!offs) throw new TiffError('A page has no image data offsets.');
  const across = Math.ceil(W / cw), down = Math.ceil(H / ch), perPlane = across * down;
  const opts = { compression: (t[259] || [1])[0], predictor: (t[317] || [1])[0], le, bps };
  const Arr = ARRAYS[dtype], planes = Array.from({ length: spp }, () => new Arr(W * H));
  const read = { uint8: 'getUint8', int8: 'getInt8', uint16: 'getUint16', int16: 'getInt16', uint32: 'getUint32',
    int32: 'getInt32', float32: 'getFloat32', float64: 'getFloat64' }[dtype];
  for (let k = 0; k < offs.length; k++) {
    const sample = planar ? Math.floor(k / perPlane) : 0, idx = planar ? k % perPlane : k;
    if (sample >= spp) break;
    const x0 = (idx % across) * cw, y0 = Math.floor(idx / across) * ch;
    const rows = tiled ? ch : Math.min(ch, H - y0), sppChunk = planar ? 1 : spp;
    const expected = cw * rows * sppChunk * bps;
    const raw = new Uint8Array(buf, offs[k], Math.min(counts ? counts[k] : expected, buf.byteLength - offs[k]));
    const bytes = await decodeChunk(raw, { ...opts, cols: cw, rows, spp: sppChunk, expected });
    const cdv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let r = 0; r < rows && y0 + r < H; r++) {
      for (let c = 0; c < cw && x0 + c < W; c++) {
        const dst = (y0 + r) * W + x0 + c, src = (r * cw + c) * sppChunk * bps;
        if (planar) planes[sample][dst] = cdv[read](src, le);
        else for (let s = 0; s < spp; s++) planes[s][dst] = cdv[read](src + s * bps, le);
      }
    }
  }
  return { planes, dtype, bits, spp };
}

function parseImageJ(desc) {
  const kv = Object.fromEntries(desc.split('\n').map(l => l.split('=')).filter(p => p.length === 2)
    .map(([k, v]) => [k.trim(), v.trim()]));
  return { C: +kv.channels || 1, Z: +kv.slices || 1, T: +kv.frames || 1, spacing: +kv.spacing || 0,
    unit: kv.unit, finterval: +kv.finterval || 0 };
}

function parseOme(xml) {
  const px = xml.match(/<(?:\w+:)?Pixels\b[^>]*>/);
  if (!px) return null;
  const a = k => { const m = px[0].match(new RegExp(`\\b${k}="([^"]*)"`)); return m ? m[1] : null; };
  const um = (v, u) => (v == null ? 0 : +v * (UNIT_UM[u || 'µm'] ?? 1));
  return { C: +a('SizeC') || 1, Z: +a('SizeZ') || 1, T: +a('SizeT') || 1, order: a('DimensionOrder') || 'XYZCT',
    dx: um(a('PhysicalSizeX'), a('PhysicalSizeXUnit')), dz: um(a('PhysicalSizeZ'), a('PhysicalSizeZUnit')),
    dt: +(a('TimeIncrement') || 0) };
}

// Page axes, slowest first, as [letter, size]; letters other than Z and C count as time.
function pageAxes(desc, nPages, spp) {
  if (desc.startsWith('ImageJ=')) {
    const ij = parseImageJ(desc), C = spp > 1 ? 1 : ij.C;  // RGB ImageJ files keep colour in the samples
    return { axes: [['T', ij.T], ['Z', ij.Z], ['C', C]], source: 'ImageJ', ij };
  }
  if (desc.includes('<OME')) {
    const o = parseOme(desc);
    if (o) {
      const size = { T: o.T, Z: o.Z, C: spp > 1 ? Math.max(1, o.C / spp) : o.C };
      return { axes: [...o.order.slice(2)].reverse().map(l => [l, size[l]]), source: 'OME-TIFF', ome: o };
    }
  }
  if (desc.startsWith('{')) {
    try {
      const d = JSON.parse(desc), shape = d.shape || [], axes = d.axes || '';
      if (axes.length === shape.length) {
        const lead = [...axes].map((l, i) => [l, shape[i]]).filter(([l]) => !'YXS'.includes(l));
        return { axes: lead, source: `tifffile ${axes}` };
      }
    } catch { /* not JSON: fall through */ }
  }
  return { axes: [['T', nPages]], source: 'plain stack' };
}

// Read a TIFF from an ArrayBuffer. Result: { W, H, T, Z, C, dtype, bits, planes, plane(t, z, c), um_per_px, z_um,
// frame_s, scaleKnown, scaleSource, layout, pages, description }.
export async function readTiff(buf) {
  if (buf.byteLength < 8) throw new TiffError('Not a TIFF file.');
  const dv = new DataView(buf), bo = dv.getUint16(0);
  if (bo !== 0x4949 && bo !== 0x4d4d) throw new TiffError('Not a TIFF file.');
  const le = bo === 0x4949, magic = dv.getUint16(2, le), big = magic === 43;
  if (magic !== 42 && !big) throw new TiffError('Not a TIFF file.');
  const ifds = [];
  for (let off = big ? Number(dv.getBigUint64(8, le)) : dv.getUint32(4, le), g = 0; off && g < 200000; g++) {
    if (off >= buf.byteLength) break;
    const { tags, next } = readIfd(dv, off, le, big);
    ifds.push(tags); off = next;
  }
  if (!ifds.length) throw new TiffError('No images in the file.');
  const first = ifds[0], W = first[256][0], H = first[257][0];
  const pages = ifds.filter(t => t[256] && t[256][0] === W && t[257][0] === H && !((t[254] || [0])[0] & 1));
  const desc = typeof first[270] === 'string' ? first[270] : '';
  const decoded = [];
  for (const t of pages) decoded.push(await readPage(buf, dv, t, le));
  const { dtype, bits, spp } = decoded[0];
  if (decoded.some(d => d.dtype !== dtype || d.spp !== spp)) throw new TiffError('Pages differ in sample type.');
  let { axes, source, ij, ome } = pageAxes(desc, pages.length, spp);
  if (axes.reduce((p, [, n]) => p * n, 1) !== pages.length) { axes = [['T', pages.length]]; source += ' (page count disagrees: read as a plain stack)'; }
  const size = { T: 1, Z: 1, C: 1 };
  for (const [l, n] of axes) size[l === 'Z' || l === 'C' ? l : 'T'] *= n;
  const C = size.C * spp, Z = size.Z, T = size.T, planes = new Array(T * Z * C);
  decoded.forEach((d, p) => {
    const idx = { T: 0, Z: 0, C: 0 };
    let rest = p;
    for (let i = axes.length - 1; i >= 0; i--) {  // row-major: last axis fastest
      const [l, n] = axes[i], g = l === 'Z' || l === 'C' ? l : 'T', v = rest % n;
      rest = Math.floor(rest / n);
      idx[g] += v * axes.slice(i + 1).filter(([m]) => (m === 'Z' || m === 'C' ? m : 'T') === g).reduce((a, [, k]) => a * k, 1);
    }
    for (let s = 0; s < spp; s++) planes[(idx.T * Z + idx.Z) * C + idx.C * spp + s] = d.planes[s];
  });
  let um = 1, zum = 0, dt = 0, scaleKnown = false, scaleSource = 'none in the file (pixels)';
  const res = first[282] ? first[282][0] : 0, resUnit = (first[296] || [2])[0];
  if (ij) {
    const f = UNIT_UM[ij.unit];
    if (f && res > 0) { um = f / res; scaleKnown = true; scaleSource = 'ImageJ unit and XResolution'; }
    if (ij.spacing && f) zum = ij.spacing * f;
    dt = ij.finterval;
  } else if (ome) {
    if (ome.dx) { um = ome.dx; scaleKnown = true; scaleSource = 'OME PhysicalSizeX'; }
    zum = ome.dz; dt = ome.dt;
  }
  if (!scaleKnown && res > 0 && resUnit === 3) { um = 1e4 / res; scaleKnown = true; scaleSource = 'XResolution (pixels per cm)'; }
  return { W, H, T, Z, C, dtype, bits, planes, plane: (t, z, c) => planes[(t * Z + z) * C + c],
    um_per_px: um, z_um: zum, frame_s: dt, scaleKnown, scaleSource, layout: source, pages: pages.length,
    compression: (first[259] || [1])[0], description: desc.slice(0, 2000), bigtiff: big, littleEndian: le };
}

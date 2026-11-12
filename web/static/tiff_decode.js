// TIFF strip/tile decoders for the browser reader (tiff_reader.js): LZW, PackBits and Deflate (zlib, through the
// browser's DecompressionStream), and the two TIFF predictors (2: horizontal differencing, 3: floating point).
// Every function works on Uint8Arrays; tested against tifffile through tiff_reader.test.mjs.

export class TiffError extends Error {}

// TIFF LZW (MSB-first codes, 9-12 bits, "early change": the width grows one code before the table fills).
export function lzwDecode(src, expected) {
  const out = new Uint8Array(expected);
  const prefix = new Int32Array(4096), suffix = new Uint8Array(4096), length = new Int32Array(4096);
  for (let i = 0; i < 256; i++) { prefix[i] = -1; suffix[i] = i; length[i] = 1; }
  let width = 9, next = 258, prev = -1, bitPos = 0, o = 0;
  const totalBits = src.length * 8;
  const write = code => {
    const n = length[code];
    let p = o + n - 1, c = code;
    while (c >= 0) { if (p < expected) out[p] = suffix[c]; p--; c = prefix[c]; }
    o += n;
  };
  const first = code => { let c = code; while (prefix[c] >= 0) c = prefix[c]; return suffix[c]; };
  while (bitPos + width <= totalBits && o < expected) {
    let code = 0;
    for (let k = 0; k < width; k++, bitPos++) code = (code << 1) | ((src[bitPos >> 3] >> (7 - (bitPos & 7))) & 1);
    if (code === 256) { width = 9; next = 258; prev = -1; continue; }
    if (code === 257) break;
    if (prev < 0) { write(code); prev = code; continue; }
    if (code < next) {
      write(code);
      if (next < 4096) { prefix[next] = prev; suffix[next] = first(code); length[next] = length[prev] + 1; next++; }
    } else {
      if (next >= 4096) throw new TiffError('Corrupt LZW data.');
      prefix[next] = prev; suffix[next] = first(prev); length[next] = length[prev] + 1; next++;
      write(code);
    }
    prev = code;
    if (next + 1 >= (1 << width) && width < 12) width++;
  }
  return out;
}

// PackBits (Apple run-length encoding, TIFF compression 32773).
export function packbitsDecode(src, expected) {
  const out = new Uint8Array(expected);
  let i = 0, o = 0;
  while (i < src.length && o < expected) {
    const n = (src[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n && o < expected; k++) out[o++] = src[i++]; }
    else if (n !== -128) { const v = src[i++]; for (let k = 0; k < 1 - n && o < expected; k++) out[o++] = v; }
  }
  return out;
}

// zlib-wrapped Deflate (TIFF compression 8 and 32946).
export async function inflate(src) {
  if (typeof DecompressionStream === 'undefined') throw new TiffError('This browser cannot decompress Deflate TIFFs.');
  const stream = new Blob([src]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Undo predictor 2 in place: each sample is stored as the difference from the sample spp positions to its left.
export function undoHorizontal(bytes, cols, rows, spp, bps, le) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), rowBytes = cols * spp * bps;
  for (let r = 0; r < rows; r++) {
    const base = r * rowBytes;
    for (let i = spp; i < cols * spp; i++) {
      const at = base + i * bps, prev = base + (i - spp) * bps;
      if (bps === 1) bytes[at] = (bytes[at] + bytes[prev]) & 0xff;
      else if (bps === 2) dv.setUint16(at, (dv.getUint16(at, le) + dv.getUint16(prev, le)) & 0xffff, le);
      else if (bps === 4) dv.setUint32(at, (dv.getUint32(at, le) + dv.getUint32(prev, le)) >>> 0, le);
      else throw new TiffError(`Predictor 2 with ${bps * 8}-bit samples is not supported.`);
    }
  }
  return bytes;
}

// Undo predictor 3 (Adobe floating-point predictor): per row, bytes were split into planes (most significant first)
// and byte-differenced. Returns the row bytes rebuilt in the file's byte order.
export function undoFloat(bytes, cols, rows, spp, bps, le) {
  const n = cols * spp, rowBytes = n * bps, out = new Uint8Array(bytes.length);
  for (let r = 0; r < rows; r++) {
    const base = r * rowBytes;
    for (let i = spp; i < rowBytes; i++) bytes[base + i] = (bytes[base + i] + bytes[base + i - spp]) & 0xff;
    for (let s = 0; s < n; s++) {
      for (let b = 0; b < bps; b++) {
        const v = bytes[base + b * n + s];  // plane b holds byte b counted from the most significant
        out[base + s * bps + (le ? bps - 1 - b : b)] = v;
      }
    }
  }
  return out;
}

// Decompress one strip or tile and undo its predictor. `expected` is the uncompressed size in bytes.
export async function decodeChunk(raw, { compression, predictor, cols, rows, spp, bps, le, expected }) {
  let bytes;
  if (compression === 1) bytes = raw.slice(0, expected);
  else if (compression === 5) bytes = lzwDecode(raw, expected);
  else if (compression === 32773) bytes = packbitsDecode(raw, expected);
  else if (compression === 8 || compression === 32946) bytes = await inflate(raw);
  else throw new TiffError(`TIFF compression ${compression} (e.g. JPEG, ZSTD) is not supported in the browser: ` +
    'open the file in the desktop app or the local web app, or save it uncompressed / LZW / Deflate in Fiji.');
  if (bytes.length < expected) { const pad = new Uint8Array(expected); pad.set(bytes); bytes = pad; }
  if (predictor === 2) undoHorizontal(bytes, cols, rows, spp, bps, le);
  else if (predictor === 3) bytes = undoFloat(bytes, cols, rows, spp, bps, le);
  return bytes;
}

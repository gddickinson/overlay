// Typed arrays <-> the local app's JSON arrays { shape, dtype, data (base64, little-endian) }.
import { ARRAY } from './filters.js';

export function toB64(typed, shape, dtype) {
  const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { shape, dtype, data: btoa(s) };
}

export function fromB64({ data, dtype }) {
  const bin = atob(data), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new ARRAY[dtype](bytes.buffer);
}

// (T, H, W) stack from frames
export function stackB64(frames, H, W, dtype) {
  const all = new ARRAY[dtype](frames.length * H * W);
  frames.forEach((f, t) => all.set(f, t * H * W));
  return toB64(all, [frames.length, H, W], dtype);
}

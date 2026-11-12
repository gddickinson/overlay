// Web worker: runs the processing/analysis ports off the page's main thread so the view stays responsive.
// Message { id, fn, args } -> { id, result } or { id, error }.
import { FN } from './worker_fns.js';

self.onmessage = ({ data: { id, fn, args } }) => {
  try { self.postMessage({ id, result: FN[fn](...args) }); } catch (e) { self.postMessage({ id, error: e.message }); }
};

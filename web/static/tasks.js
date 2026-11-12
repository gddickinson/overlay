// Run a worker.js function off the main thread: await run('projection', frames, 'max', 'uint16').
// Falls back to running in the page if module workers are unavailable.
let worker = null, seq = 0;
const pending = new Map();

function get() {
  if (worker === null) {
    try {
      worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }) => {
        const p = pending.get(data.id);
        if (!p) return;
        pending.delete(data.id);
        if (data.error) p.reject(new Error(data.error)); else p.resolve(data.result);
      };
    } catch { worker = false; }
  }
  return worker;
}

export async function run(fn, ...args) {
  const w = get();
  if (!w) {
    const { FN } = await import('./worker_fns.js');
    return FN[fn](...args);
  }
  const id = ++seq;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); w.postMessage({ id, fn, args }); });
}

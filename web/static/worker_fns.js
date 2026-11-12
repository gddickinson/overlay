// Functions the worker (worker.js) can run; tasks.js runs them in the page when workers are unavailable.
import { applyFilter } from './filters.js';
import * as ops from './ops.js';
import { analyzeTrend, analyzePeaks } from './stats.js';

export const FN = {
  applyFilter,
  // run one filter over every frame
  filterStack: (frames, H, W, dtype, type, p) => frames.map(f => applyFilter(f, H, W, dtype, type, p)),
  projection: ops.projection, applyContrast: ops.applyContrast, rescaleFrame: ops.rescaleFrame,
  rescaleStack: (frames, lo, hi) => frames.map(f => ops.rescaleFrame(f, lo, hi)),
  equalizeHist: ops.equalizeHist, equalizeStack: (frames, dtype) => frames.map(f => ops.equalizeHist(f, dtype)),
  segment: ops.segment, segmentStack: (frames, dtype) => frames.map(f => ops.segment(f, dtype)),
  timeSeries: ops.timeSeries, basicStats: ops.basicStats, measure: ops.measure,
  analyze: (series) => ({ trend: analyzeTrend(series), peaks: analyzePeaks(series) }),
};

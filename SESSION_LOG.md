# Session log

## 2026-11-11/12: web interface, static GitHub Pages site and bug fixes (branch web-interface)
- Bug fixes (regression tests in `tests/test_regressions.py`, each failed before its fix):
  - `ImageProcessor.apply_filter('median')` silently returned the unfiltered frame (skimage's removed `selem`
    keyword); `segment_image('adaptive')` returned the threshold map instead of a mask;
    `utils.helpers.convert_colormap_to_lut` crashed on matplotlib >= 3.9.
  - Controls-panel Processing/Analysis buttons raised TypeError (wrong MainWindow signatures; aborts a PyQt6 app);
    the 'Current ROIs' choice never matched; the filter parameter was ignored.
  - Mask overlay: 16-bit frames wrapped modulo 256 and the RGB composite was shown as a 3-frame movie; the
    colormap was never applied. Blend moved to `core/overlay_blend.py`.
- `core/processing_defaults.py`: all ImageProcessor defaults in one place (behaviour unchanged).
- Web: `web/` (FastAPI app, catalogue, Python compute) and `web/static/` front end (Viewer, Process, Analyze,
  About), port 8128. Pure-JS TIFF reader tested against tifffile on 19 fixture files; JS ports of the processing
  and analysis code tested against Python (`ops.test.mjs`). Synthetic demo stacks from `scripts/make_demo_data.py`
  (no example data is committed in the repository).
- Desktop checked offscreen with the demo stacks (loads, overlay, colormap).
- Open points for George: `apply_filter('sobel'/'laplacian')` returns floats in 0-1 (or signed), which
  MainWindow.apply_filter writes back into the integer stack, so the frame becomes ~0 (needs a decision: rescale,
  or keep results in a separate float stack); the "bilateral" filter is a Gaussian of sigma_spatial;
  `detect_features` passes raw intensities to `blob_log` with threshold 0.1, so 16-bit frames give hundreds of blobs;
  `utils/config.py` `load_config` copies DEFAULT_CONFIG shallowly, so a loaded config mutates the defaults;
  the desktop treats multi-channel (T, C, Y, X) TIFFs as (T, H, W, C) wrongly. `gui/main_window.py` (1331 lines),
  `gui/dialogs.py` (1609), `gui/image_view.py`, `core/data_analyzer.py`, `core/export_manager.py` exceed 500 lines.

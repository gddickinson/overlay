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
- Open points: the "bilateral" filter is a Gaussian of sigma_spatial. `gui/main_window.py` (1350 lines),
  `gui/dialogs.py` (1609), `gui/image_view.py`, `core/data_analyzer.py`, `core/export_manager.py` exceed 500 lines.

## 2026-11-12: second round of fixes (George: "fix any other problems")
- Sobel/Laplacian from the desktop: float results kept (the stack becomes float32) instead of truncated to ~0.
- Blob detection: frame scaled to 0-1 between its 1st and 99.8th percentiles before blob_log
  (DEFAULTS['features']['blob']['norm_percentiles']); demo puncta frame 10 blobs instead of 695. Earlier blob
  counts change.
- load_config / reset_to_defaults return deep copies (a loaded config no longer changes DEFAULT_CONFIG).
- ImageStack reads ImageJ/OME hyperstacks by their axes (core/tiff_axes.py): frames = T x Z, channels selectable
  (Image > Next Channel); same layout as the browser reader (test in tests/test_web.py).

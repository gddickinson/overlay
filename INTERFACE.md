# INTERFACE.md: navigation map

Advanced TIFF Stack Viewer: a PyQt6/pyqtgraph desktop app (fluorescence stack + binary mask overlay) and a web
interface (FastAPI + plain ES modules, static GitHub Pages build).

## Entry points
- `main.py`: desktop app (`python main.py [-f fluor.tif] [-m mask.tif]`) -> `gui.main_window.MainWindow`.
- `run_web.py`: web app on http://127.0.0.1:8128 -> `web.app:app`.
- `standalone/tiff_stack_viewer.py`: minimal self-contained viewer.

## core/ (Qt-free; used by the GUI and the web API)
- `image_loader.py`: `ImageStack` (data (T, H, W[, C]), `load_tiff`, `get_frame`, `get_normalized_frame`,
  `create_z_projection`, `apply_contrast`, `resize`, `save_tiff`), `ImageLoader` (TIFF/HDF5/image series).
- `image_processor.py`: `ImageProcessor` (`apply_filter`, `enhance_contrast`, `register_images`, `segment_image`,
  `detect_features`, `overlay_images`, `measure_intensity`, `extract_time_series`). Defaults from
  `processing_defaults.py`.
- `processing_defaults.py`: `DEFAULTS[group][op][param]`, the one place for processing defaults (GUI fallbacks and
  the web's initial values).
- `overlay_blend.py`: `to_uint8`, `blend_overlay` (mask-over-fluorescence composite used by `gui/image_view.py`).
- `data_analyzer.py`: `DataAnalyzer` (basic stats, ROI time-series trend/peaks, PCA, clustering, heatmaps).
- `export_manager.py`: `ExportManager` (images, frames, movies, data, reports).

## gui/ (PyQt6)
- `main_window.py`: `MainWindow` (menus, docks, loading, display update, dialogs, processing/analysis actions).
- `image_view.py`: `EnhancedImageView` (pyqtgraph view, overlay via `core.overlay_blend`, ROIs, features).
- `controls_panel.py`: `ControlsPanel` (Display/Processing/Analysis tabs; forwards to MainWindow methods).
- `dialog_presets.py`: pre-fills dialogs with the controls panel's values.
- `dialogs.py`: settings, contrast, projection, registration, analysis, export, results, time-series, log dialogs.
- `widgets/`: histogram, ROI toolbar, navigation bar, timeline.

## utils/
- `config.py` (`DEFAULT_CONFIG`, load/save), `logger.py` (`setup_logger`), `helpers.py` (`convert_colormap_to_lut`, ...).

## web/
- `app.py`: routes `/api/info`, `/api/luts`, `/api/ops`, `/api/demos` (exported as JSON for the static site,
  `STATIC_ENDPOINTS`), `POST /api/process`, `POST /api/register` (local app only).
- `catalog.py`: LUTs (via `utils.helpers`), mask colours/colormaps (mirroring `gui/controls_panel.py`), display
  defaults (`utils.config`), operation specs (defaults from `core.processing_defaults`), demo list.
- `compute.py`: base64 array codec and `process`/`register` calling `ImageProcessor`.
- `static/`: `index.html`, `style.css` + `core.js` (shared kit), `style_app.css`, `main.js` (router setup),
  `static_api.js` (static-site data layer), pages `home.js`, `viewer.js` (+ `viewer_panel.js`), `process.js`,
  `analyze.js`, `about.js`; data/engine modules `session.js` (loaded stacks), `render.js` (composite, overlays,
  histogram), `charts.js`, `b64.js`, `tasks.js` + `worker.js` + `worker_fns.js` (Web Worker);
  `tiff_reader.js` + `tiff_decode.js` (TIFF reader), `filters.js`, `ops.js`, `stats.js` (ports of the Python
  operations); node tests `tiff_reader.test.mjs`, `ops.test.mjs`; `demo/` (synthetic demo stacks).

## scripts/
- `make_demo_data.py` (demo stacks), `make_tiff_fixtures.py` (TIFF test files + expected values from tifffile),
  `make_ops_fixture.py` (Python reference values for the JS ports), `build_static.py` (Pages export),
  `screenshots.py` (Playwright smoke test / README images).

## tests/
- `test_web.py` (API, static build, node tests, fixtures current), `test_regressions.py` (fixed bugs; Qt parts
  offscreen), `fixtures/` (TIFF files, `tiff_expected.json`, `ops_cases.json`).

## How modules connect
GUI -> core (+ utils). Web API -> core + utils (never gui). Browser -> `session.js` (data) -> `render.js` (display);
`process.js`/`analyze.js` -> `tasks.js` -> worker -> `filters.js`/`ops.js`/`stats.js`; Python-only operations ->
`/api/process` -> `web/compute.py` -> `ImageProcessor`.

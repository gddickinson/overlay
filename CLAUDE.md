# overlay (Advanced TIFF Stack Viewer)

@INTERFACE.md

- Desktop: `python main.py`. Web: `python run_web.py` (port 8128). Tests: `python -m pytest -q tests` (Qt tests
  run offscreen; node runs the JS tests).
- Processing defaults live in `core/processing_defaults.py` only. After changing processing code, regenerate
  `python scripts/make_ops_fixture.py` and check `node web/static/ops.test.mjs`.
- Keep files under 500 lines (`gui/main_window.py`, `gui/dialogs.py`, `gui/image_view.py`, `core/data_analyzer.py`
  and `core/export_manager.py` are older files already over budget). Session notes go to SESSION_LOG.md.

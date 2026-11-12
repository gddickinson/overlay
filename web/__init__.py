"""Web interface for the Advanced TIFF Stack Viewer (FastAPI app + plain ES-module front end).

- app.py: routes under /api and STATIC_ENDPOINTS (exported to JSON for the static site)
- catalog.py: LUTs, mask colours, operation specs (defaults from core/processing_defaults.py) and demo files
- compute.py: Python-side processing for the local app (frames in base64, ImageProcessor operations)
- static/: the front end (TIFF reader, JS ports of the processing code, pages); static/demo/ holds the demo stacks
"""

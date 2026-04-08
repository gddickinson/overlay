# Advanced TIFF Stack Viewer (Overlay) — Roadmap

## Current State
A well-structured PyQt6/pyqtgraph application for fluorescence microscopy analysis. Clean modular architecture: `core/` (image_loader, image_processor, data_analyzer, export_manager), `gui/` (main_window, image_view, controls_panel, dialogs, widgets/), `utils/` (config, logger, helpers). Supports ROI tools, time-series analysis, z-projection, PCA/clustering, registration, and CLI loading. Has `requirements.txt`, `example_data/` with real TIFF stacks, and a `standalone/` tools directory. This is one of the more mature projects.

## Short-term Improvements
- [ ] Add a `pyproject.toml` for modern Python packaging alongside the existing `requirements.txt`
- [ ] Add keyboard shortcut overlay/help dialog listing all navigation and ROI shortcuts
- [ ] Improve error messages in `core/image_loader.py` for corrupted or unsupported TIFF formats
- [ ] Add undo/redo for ROI placement and deletion in `gui/widgets/roi_tools.py`
- [ ] Add tooltips to all controls in `gui/controls_panel.py` explaining what each adjustment does
- [ ] Validate frame indices in `gui/widgets/timeline_widget.py` to prevent out-of-bounds navigation

## Feature Enhancements
- [ ] Add multi-channel overlay support (not just fluorescence + mask, but 3+ channels with individual colormaps)
- [ ] Add line profile tool — draw a line and plot intensity along it
- [ ] Support ND2 (Nikon), CZI (Zeiss), and LIF (Leica) file formats via `aicsimageio`
- [ ] Add particle detection and counting overlay using scikit-image blob detection
- [ ] Add kymograph generation from a line ROI across time frames
- [ ] Export ROI measurements directly to Excel format with charts
- [ ] Add a movie export with timestamp and scale bar overlay

## Long-term Vision
- [ ] Add napari integration as an alternative viewer backend for 3D rendering
- [ ] Support remote file loading (S3, OMERO server) for lab-scale deployments
- [ ] Add a plugin system for custom analysis pipelines (similar to ImageJ/FIJI)
- [ ] Implement GPU-accelerated image processing for large stacks (>10GB)
- [ ] Add collaborative annotation: export/import ROI sets in a standardized format (OME-ROI)
- [ ] Package as a standalone app with PyInstaller for distribution to non-Python users

## Technical Debt
- [ ] Add unit tests for `core/image_processor.py` filter and projection functions
- [ ] Add integration tests loading the example TIFF stacks and verifying basic operations
- [ ] Review `gui/dialogs.py` — if it contains many dialog classes, split into `dialogs/` subpackage
- [ ] Document the `standalone/` scripts and clarify their relationship to the main app
- [ ] Add type hints to callback functions in `gui/main_window.py` signal connections
- [ ] Profile memory usage with large stacks and add memory-mapped loading for files >2GB

"""Regression tests for bugs fixed on the web-interface branch (each test failed before its fix).

Run: python -m pytest -q tests/test_regressions.py   (Qt parts run offscreen)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.image_processor import ImageProcessor  # noqa: E402


def _blobs():
    yy, xx = np.mgrid[:40, :48]
    img = 100 + 900 * np.exp(-((yy - 15) ** 2 + (xx - 20) ** 2) / 40.0)
    img[5, 5] = 4000  # a hot pixel a median filter must remove
    return img.astype(np.uint16)


def test_median_filter_is_applied():
    """apply_filter('median') passed skimage's removed `selem` keyword, so it failed and silently returned the input."""
    img = _blobs()
    out = ImageProcessor().apply_filter(img, "median", size=1)
    assert out.dtype == img.dtype
    assert out[5, 5] < 1000, "hot pixel survived: the median filter did not run"


def test_adaptive_segmentation_returns_a_mask():
    """segment_image(method='adaptive') returned skimage's threshold map instead of a boolean mask."""
    img = _blobs().astype(float)
    out = ImageProcessor().segment_image(img, method="adaptive", block_size=15)
    assert out.dtype == bool
    assert out.shape == img.shape and out[15, 20] and out.mean() < 0.5


def test_colormap_lut_on_current_matplotlib():
    """convert_colormap_to_lut called matplotlib.cm.get_cmap, removed in matplotlib 3.9."""
    from utils.helpers import convert_colormap_to_lut
    lut = convert_colormap_to_lut("viridis")
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert tuple(lut[0]) == (68, 1, 84)


# --- desktop GUI --------------------------------------------------------------------------------------------------
@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("HOME", str(tmp_path))  # the app writes logs and config under ~/.tiff_stack_viewer
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication.instance() or QApplication([])
    import pyqtgraph as pg
    pg.setConfigOptions(imageAxisOrder="row-major")
    monkeypatch.setattr(QDialog, "exec", lambda self: 0)  # never block on a modal dialog
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    from utils.config import DEFAULT_CONFIG
    from gui.main_window import MainWindow
    from core.image_loader import ImageStack
    import copy
    w = MainWindow(copy.deepcopy(DEFAULT_CONFIG))
    st = ImageStack()
    st.data = np.stack([_blobs()] * 3)
    st._update_dimensions()
    w.fluorescence_stack = st
    mk = ImageStack()
    mk.data = (st.data > 500).astype(np.uint8)
    mk._update_dimensions()
    w.mask_stack = mk
    yield w
    w.close()
    app.processEvents()


def test_controls_panel_buttons_reach_the_main_window(window):
    """The Processing/Analysis tab buttons called main-window methods with arguments they did not accept
    (TypeError, which aborts a PyQt6 app), and 'Current ROIs' was compared after lower-casing so never matched."""
    panel = window.controls_panel
    panel.filter_combo.setCurrentText("Gaussian")
    panel.filter_param_spin.setValue(2.0)
    before = window.fluorescence_stack.data[0].copy()
    panel.apply_filter()
    assert not np.array_equal(before, window.fluorescence_stack.data[0])
    panel.apply_registration()
    panel.create_projection()
    for analysis in ("Intensity", "Time Series"):
        panel.roi_analysis_combo.setCurrentText(analysis)
        for source in ("Current ROIs", "Mask", "Whole Image"):
            panel.roi_source_combo.setCurrentText(source)
            panel.analyze_roi()
    panel.detect_features()


def test_controls_panel_filter_parameter_is_used(window):
    """The panel's filter parameter was dropped: Gaussian always ran with sigma 1."""
    panel = window.controls_panel
    panel.filter_combo.setCurrentText("Gaussian")
    panel.filter_param_spin.setValue(3.0)
    raw = window.fluorescence_stack.data[0].copy()
    panel.apply_filter()
    expected = ImageProcessor().apply_filter(raw, "gaussian", sigma=3.0)
    assert np.array_equal(window.fluorescence_stack.data[0], expected)


def test_mask_overlay_keeps_16_bit_images(window):
    """With a mask shown, a 16-bit frame was cast to uint8 (values wrapped modulo 256) before display, and the RGB
    composite was passed to ImageView as (3, H, W), which it reads as a 3-frame movie (only the red plane showed)."""
    window.update_display()
    shown = window.image_view.get_current_image()
    assert shown.shape == window.fluorescence_stack.data[0].shape + (3,)
    raw = window.fluorescence_stack.data[0]
    off = window.mask_stack.data[0] == 0
    # outside the mask the display must be a monotonic function of intensity
    order = np.argsort(raw[off], kind="stable")
    assert np.all(np.diff(shown[..., 0][off][order].astype(int)) >= 0)


def test_colormap_applies_without_mask(window):
    """The colormap choice was never applied: every frame was stacked to RGB before the 2D-only LUT step."""
    window.display_settings["mask_visible"] = False
    window.display_settings["colormap"] = "viridis"
    window.update_display()
    assert window.image_view.view.getImageItem().lut is not None

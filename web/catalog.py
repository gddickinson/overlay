"""Catalogue data served to the front end: LUTs, mask colours, operation specs and demo files.

Everything here comes from the repository's own code: colormap names and mask colours from the desktop controls
panel (gui/controls_panel.py; checked by tests/test_web.py), LUT tables from utils.helpers.convert_colormap_to_lut,
operation defaults from core/processing_defaults.DEFAULTS, demo files from scripts/make_demo_data.py.
"""
import inspect

from core.image_loader import ImageStack
from core.processing_defaults import DEFAULTS
from utils.config import DEFAULT_CONFIG
from utils.helpers import convert_colormap_to_lut

# Display tab of the desktop controls panel: "Colormap" combo and "Mask Color" combo (same order)
DESKTOP_COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "hot", "cool", "jet", "rainbow"]
MASK_COLORS = {"Red": [255, 0, 0], "Green": [0, 255, 0], "Blue": [0, 0, 255],
               "Yellow": [255, 255, 0], "Magenta": [255, 0, 255], "Cyan": [0, 255, 255]}


def display_defaults():
    """The desktop's display/navigation defaults (utils/config.py DEFAULT_CONFIG); auto levels use the 1st-99th
    percentiles, as gui/image_view.py _get_display_levels does."""
    d, nav = DEFAULT_CONFIG["display"], DEFAULT_CONFIG["navigation"]
    return {"mask_color": d["default_mask_color"], "overlay_alpha": d["default_overlay_alpha"],
            "colormap": d["colormap"], "show_scale_bar": d["show_scale_bar"], "frame_rate": nav["frame_rate"],
            "loop_playback": nav["loop_playback"], "auto_levels_percentiles": [1, 99]}


def luts():
    """256-entry RGB tables: single-colour ramps (black to each mask colour) and the desktop colormaps."""
    out = {name: [[round(c * i / 255) for c in rgb] for i in range(256)] for name, rgb in MASK_COLORS.items()}
    out["Grays"] = [[i, i, i] for i in range(256)]
    for name in DESKTOP_COLORMAPS:
        out[name] = convert_colormap_to_lut(name).tolist()
    return out


def _p(group, op, key, label, lo, hi, step):
    return {"key": key, "label": label, "min": lo, "max": hi, "step": step, "default": DEFAULTS[group][op][key]}


_CONTRAST = inspect.signature(ImageStack.apply_contrast).parameters  # the desktop Contrast dialog's stack stretch


def _sig(key, label, lo, hi, step):
    return {"key": key, "label": label, "min": lo, "max": hi, "step": step, "default": _CONTRAST[key].default}


# where: "js" runs in the browser (a JS port tested against the Python code), "python" needs the local app
OPERATIONS = [
    {"id": "gaussian", "group": "filter", "label": "Gaussian filter", "where": "js",
     "params": [_p("filter", "gaussian", "sigma", "sigma (px)", 0.1, 10, 0.1)]},
    {"id": "median", "group": "filter", "label": "Median filter (disk)", "where": "js",
     "params": [_p("filter", "median", "size", "disk radius (px)", 1, 10, 1)]},
    {"id": "bilateral", "group": "filter", "label": "'Bilateral' (the code runs a Gaussian of sigma_spatial)",
     "where": "js", "params": [_p("filter", "bilateral", "sigma_spatial", "sigma_spatial (px)", 0.1, 10, 0.1)]},
    {"id": "sobel", "group": "filter", "label": "Sobel edges", "where": "js", "params": []},
    {"id": "laplacian", "group": "filter", "label": "Laplacian", "where": "js", "params": []},
    {"id": "rescale", "group": "contrast", "label": "Contrast stretch (percentiles, this frame)", "where": "js",
     "params": [_p("contrast", "rescale", "p_low", "low percentile", 0, 50, 0.5),
                _p("contrast", "rescale", "p_high", "high percentile", 50, 100, 0.5)]},
    {"id": "stack_contrast", "group": "contrast", "label": "Contrast stretch (whole stack, ImageStack.apply_contrast)",
     "where": "js", "stack_only": True,
     "params": [_sig("p_low", "low percentile", 0, 50, 0.5), _sig("p_high", "high percentile", 50, 100, 0.5)]},
    {"id": "equalize", "group": "contrast", "label": "Histogram equalisation", "where": "js", "params": []},
    {"id": "threshold", "group": "segment", "label": "Otsu threshold -> mask", "where": "js", "params": []},
    {"id": "frangi", "group": "filter", "label": "Frangi vesselness", "where": "python", "params": []},
    {"id": "adaptive_contrast", "group": "contrast", "label": "Adaptive equalisation (CLAHE)", "where": "python",
     "params": [_p("contrast", "adaptive", "kernel_size", "kernel size (px)", 8, 256, 1),
                _p("contrast", "adaptive", "clip_limit", "clip limit", 0.001, 1, 0.001)]},
    {"id": "adaptive_threshold", "group": "segment", "label": "Local (adaptive) threshold -> mask",
     "where": "python", "params": [_p("segment", "adaptive", "block_size", "block size (odd, px)", 3, 201, 2),
                                   _p("segment", "adaptive", "offset", "offset", -1000, 1000, 1)]},
    {"id": "blob", "group": "features", "where": "python",
     "label": "Blob detection (Laplacian of Gaussian; frame scaled 0-1 between its {} and {} percentiles)".format(
         *DEFAULTS["features"]["blob"]["norm_percentiles"]),
     "frame_only": True,
     "params": [_p("features", "blob", "min_sigma", "min sigma", 1, 20, 0.5),
                _p("features", "blob", "max_sigma", "max sigma", 2, 50, 0.5),
                _p("features", "blob", "threshold", "threshold", 0.001, 1, 0.001)]},
    {"id": "corner", "group": "features", "label": "Corner detection (Harris)", "where": "python",
     "frame_only": True,
     "params": [_p("features", "corner", "min_distance", "min distance (px)", 1, 50, 1),
                _p("features", "corner", "threshold", "relative threshold", 0.01, 1, 0.01)]},
    {"id": "register", "group": "registration", "label": "Register stack to a frame (phase correlation)",
     "where": "python", "stack_only": True,
     "params": [_p("registration", "phase", "upsample_factor", "upsample factor", 1, 100, 1)]},
]

DEMOS = [
    {"id": "cytoplasm", "file": "demo/demo_cytoplasm.tif", "role": "channel", "name": "cytoplasm", "lut": "Green",
     "text": "24 frames, 128 x 128, uint16: six cells that slowly bleach (1 % per frame), camera offset and shot noise."},
    {"id": "puncta", "file": "demo/demo_puncta.tif", "role": "channel", "name": "puncta", "lut": "Magenta",
     "text": "Same size: flickering spots inside the cells over a dim cytoplasmic background."},
    {"id": "mask", "file": "demo/demo_mask.tif", "role": "mask", "name": "mask",
     "text": "Deflate-compressed uint8 mask made by the repo's ImageProcessor: Gaussian (sigma 2) then Otsu threshold."},
]
DEMO_NOTE = ("Synthetic data drawn by scripts/make_demo_data.py (seed 7), not a real recording. "
             "The 0.2 um pixel size written into the files is made up so the scale bar has something to show.")

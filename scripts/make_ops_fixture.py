"""Write tests/fixtures/ops_cases.json: the repository's own processing and analysis code run on a crop of the demo
stacks, for web/static/ops.test.mjs to check the JavaScript ports against.

Inputs: frames 0-23, rows 40-71, columns 30-69 of web/static/demo/demo_cytoplasm.tif and demo_mask.tif (written by
scripts/make_demo_data.py). Every output comes from ImageStack, ImageProcessor, DataAnalyzer or core/overlay_blend.

Run: python scripts/make_ops_fixture.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data_analyzer import DataAnalyzer  # noqa: E402
from core.image_loader import ImageStack  # noqa: E402
from core.image_processor import ImageProcessor  # noqa: E402
from core.overlay_blend import blend_overlay  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "ops_cases.json"
DEMO = ROOT / "web" / "static" / "demo"
CROP = (slice(None), slice(40, 72), slice(30, 70))


def flat(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float).ravel()]


def num(v):
    return None if v is None else float(v)


def stack_of(data):
    st = ImageStack()
    st.data = data
    st._update_dimensions()
    return st


def build():
    logging.disable(logging.CRITICAL)
    data = tifffile.imread(DEMO / "demo_cytoplasm.tif")[CROP]
    mask = tifffile.imread(DEMO / "demo_mask.tif")[CROP]
    proc, an = ImageProcessor(), DataAnalyzer()
    frame, T, H, W = data[0], *data.shape
    cases = {"input": {"T": T, "H": H, "W": W, "dtype": str(data.dtype), "data": flat(data), "mask": flat(mask)}}
    st = stack_of(data.copy())
    cases["projection"] = {m: flat(st.create_z_projection(m)) for m in ("max", "min", "mean", "median", "std")}
    cases["normalized"] = flat(st.get_normalized_frame(0))
    st2 = stack_of(data.copy())
    st2.apply_contrast(2, 98)
    cases["apply_contrast"] = {"p_low": 2, "p_high": 98, "dtype": str(st2.data.dtype), "frame0": flat(st2.data[0])}
    cases["enhance_rescale"] = flat(proc.enhance_contrast(frame, "rescale", p_low=2, p_high=98))
    cases["equalize"] = flat(proc.enhance_contrast(frame, "equalize"))
    cases["equalize_float"] = flat(proc.enhance_contrast(frame.astype(np.float32) / 7.0, "equalize"))
    cases["filters"] = {
        "gaussian_1": flat(proc.apply_filter(frame, "gaussian", sigma=1.0)),
        "gaussian_2.5": flat(proc.apply_filter(frame, "gaussian", sigma=2.5)),
        "median_1": flat(proc.apply_filter(frame, "median", size=1)),
        "median_3": flat(proc.apply_filter(frame, "median", size=3)),
        "bilateral": flat(proc.apply_filter(frame, "bilateral", sigma_spatial=2)),
        "sobel": flat(proc.apply_filter(frame, "sobel")),
        "laplacian": flat(proc.apply_filter(frame, "laplacian")),
        "sobel_u8": flat(proc.apply_filter(mask[0], "sobel")),
        "gaussian_f32": flat(proc.apply_filter(frame.astype(np.float32) / 3.0, "gaussian", sigma=1.5)),
    }
    from skimage import filters as skf
    cases["otsu"] = {"threshold": float(skf.threshold_otsu(frame)), "mask": flat(proc.segment_image(frame)),
                     "threshold_f32": float(skf.threshold_otsu(frame.astype(np.float32) / 3.0))}
    regions = [{"type": "rectangle", "points": (4, 6, 20, 30), "id": "rect"},
               {"type": "circle", "points": (16, 20, 7.5), "id": "circle"}]
    stat = lambda s: {k: num(s[k]) for k in ("mean", "std", "min", "max", "sum", "count")}  # noqa: E731
    cases["measure"] = {"whole": stat(proc.measure_intensity(frame)),
                        "mask": stat(proc.measure_intensity(frame, mask=mask[0])),
                        "regions": [stat(s) for s in proc.measure_intensity(frame, regions=regions)],
                        "region_defs": [{**r, "points": list(r["points"])} for r in regions]}
    ts = proc.extract_time_series(st, mask=mask[0])
    cases["time_series_mask"] = {k: flat(ts[k]) for k in ("mean", "std", "sum", "max", "min")}
    bs = an.calculate_basic_stats(st)
    cases["basic_stats"] = {k: flat(bs[k]) for k in ("mean", "median", "std", "min", "max", "snr")}
    tr = an._analyze_trend(ts["mean"])
    cases["trend"] = {k: (tr[k] if k == "trend" else float(tr[k])) for k in tr}
    puncta = tifffile.imread(DEMO / "demo_puncta.tif")[CROP]
    pts = proc.extract_time_series(stack_of(puncta), mask=mask[0])["mean"]
    cases["peaks_input"] = flat(pts)
    cases["peaks"] = an._find_peaks(pts)
    cases["trend_puncta"] = {k: (v if k == "trend" else float(v)) for k, v in an._analyze_trend(pts).items()}
    cases["blend"] = {"alpha": 0.4, "color": [255, 0, 255],
                      "rgb": flat(blend_overlay(frame, mask[0], 0.4, [255, 0, 255]))}
    return cases


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

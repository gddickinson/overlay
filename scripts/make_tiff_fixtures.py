"""Write the TIFF test files for the browser reader and what tifffile reads from them.

Output: tests/fixtures/tiff/*.tif and tests/fixtures/tiff_expected.json. web/static/tiff_reader.test.mjs opens every
file with the pure-JS reader and checks shape, dtype, scale and pixel checksums against the JSON; tests/test_web.py
checks that tifffile still reads the committed files to the same numbers.

Pixel data are arranged as (T, Z, C, Y, X): T collects time and any other page axis (I, Q), C collects channels and
samples (S). Checksums over that order: sum, sum of squares and sum of value * (index % 97 + 1) (catches order errors).

Needs imagecodecs for the LZW and floating-point-predictor files. Run: python scripts/make_tiff_fixtures.py
"""
import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tiff"
EXPECTED = ROOT / "tests" / "fixtures" / "tiff_expected.json"


def ramp(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "ui" else None
    if info is None:
        return (rng.standard_normal(shape) * 100).astype(dtype)
    lo, hi = max(info.min, -30000), min(info.max, 60000)
    return rng.integers(lo, hi, size=shape, endpoint=True).astype(dtype)


def cases():
    """(file name, array, imwrite keyword arguments, note)."""
    return [
        ("u8_stack.tif", ramp((5, 20, 24), np.uint8, 1), {}, "uint8, 5 pages, tifffile shaped description"),
        ("u16_bigendian.tif", ramp((4, 20, 24), np.uint16, 2), {"byteorder": ">"}, "uint16, big-endian, 4 samples per pixel (tifffile stores a leading 3 or 4 as samples)"),
        ("i16.tif", ramp((2, 20, 24), np.int16, 3), {}, "int16 with negative values"),
        ("u32.tif", ramp((2, 12, 10), np.uint32, 4), {}, "uint32"),
        ("f32.tif", ramp((5, 20, 24), np.float32, 5), {}, "float32"),
        ("f64_single.tif", ramp((20, 24), np.float64, 6), {}, "float64, one page"),
        ("rgb_contig.tif", ramp((2, 20, 24, 3), np.uint8, 7), {"photometric": "rgb"}, "RGB, chunky (contig) samples"),
        ("planar_separate.tif", ramp((3, 20, 24), np.uint16, 8),
         {"photometric": "minisblack", "planarconfig": "separate"}, "3 samples per pixel, planar (separate)"),
        ("multistrip.tif", ramp((2, 37, 24), np.uint16, 9), {"rowsperstrip": 5}, "uint16, 8 strips per page"),
        ("tiled.tif", ramp((2, 40, 56), np.uint16, 10), {"tile": (16, 16)}, "uint16, 16 x 16 tiles, partial tiles"),
        ("deflate.tif", ramp((5, 20, 24), np.uint16, 11), {"compression": "zlib"}, "Deflate"),
        ("deflate_pred.tif", ramp((5, 20, 24), np.uint16, 12), {"compression": "zlib", "predictor": True},
         "Deflate + horizontal predictor"),
        ("lzw_pred.tif", ramp((5, 20, 24), np.uint16, 13), {"compression": "lzw", "predictor": True},
         "LZW + horizontal predictor"),
        ("lzw_u8.tif", ramp((2, 33, 47), np.uint8, 14), {"compression": "lzw"}, "LZW, uint8, odd sizes"),
        ("packbits.tif", (ramp((2, 20, 24), np.uint8, 15) // 64 * 64).astype(np.uint8), {"compression": "packbits"},
         "PackBits (runs)"),
        ("float_pred.tif", ramp((2, 20, 24), np.float32, 16), {"compression": "zlib", "predictor": True},
         "float32, Deflate + floating-point predictor"),
        ("imagej_hyper.tif", ramp((3, 4, 2, 20, 24), np.uint16, 17),
         {"imagej": True, "resolution": (1 / 0.16, 1 / 0.16),
          "metadata": {"axes": "TZCYX", "unit": "um", "spacing": 0.5, "finterval": 2.0}},
         "ImageJ hyperstack T=3 Z=4 C=2, 0.16 um pixels, 0.5 um spacing, 2 s interval"),
        ("ome.ome.tif", ramp((3, 2, 20, 24), np.uint16, 18),
         {"ome": True, "metadata": {"axes": "TCYX", "PhysicalSizeX": 0.108, "PhysicalSizeXUnit": "µm",
                                    "PhysicalSizeY": 0.108, "PhysicalSizeYUnit": "µm", "TimeIncrement": 0.05}},
         "OME-TIFF T=3 C=2, 0.108 um pixels"),
        ("bigtiff.tif", ramp((2, 20, 24), np.uint16, 19), {"bigtiff": True}, "BigTIFF"),
    ]


def canonical(path):
    """(T, Z, C, Y, X) array as tifffile reads the first series."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr, axes = s.asarray(), s.axes
    groups = {"T": [], "Z": [], "C": []}
    for i, a in enumerate(axes):
        if a in "YX":
            continue
        groups["Z" if a == "Z" else "C" if a in "CS" else "T"].append(i)
    order = groups["T"] + groups["Z"] + groups["C"] + [axes.index("Y"), axes.index("X")]
    arr = arr.transpose(order)
    dims = [int(np.prod([s.shape[i] for i in groups[k]])) for k in "TZC"]
    return arr.reshape(*dims, arr.shape[-2], arr.shape[-1]), axes, str(arr.dtype)


def checksums(arr):
    v = arr.astype(np.float64).ravel()
    w = (np.arange(v.size) % 97 + 1).astype(np.float64)
    return {"sum": float(v.sum()), "sumsq": float((v * v).sum()), "weighted": float((v * w).sum()),
            "first": [float(x) for x in v[:8]], "min": float(v.min()), "max": float(v.max())}


def expected_for(path, note=""):
    arr, axes, dtype = canonical(path)
    T, Z, C, H, W = arr.shape
    return {"note": note, "axes": axes, "dtype": dtype, "T": T, "Z": Z, "C": C, "H": H, "W": W, **checksums(arr)}


SCALES = {"imagej_hyper.tif": {"um_per_px": 0.16, "z_um": 0.5, "frame_s": 2.0},
          "ome.ome.tif": {"um_per_px": 0.108, "frame_s": 0.05}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, arr, kw, note in cases():
        tifffile.imwrite(OUT / name, arr, **kw)
        expected[name] = {**expected_for(OUT / name, note), **SCALES.get(name, {})}
    EXPECTED.write_text(json.dumps(expected, indent=1) + "\n")
    print(f"{len(expected)} TIFF fixtures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

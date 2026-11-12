"""Write the synthetic demo stacks used by the web interface (web/static/demo/).

Nothing here is real data. The stacks are drawn from a seeded random generator so the web viewer has something to open:

- demo_cytoplasm.tif: 24 frames x 128 x 128, uint16, ImageJ format, uncompressed. Six elliptical "cells" with soft
  edges that slowly bleach (1 % per frame).
- demo_puncta.tif: same shape. Small Gaussian spots inside the cells whose brightness flickers from frame to frame,
  over a dim copy of the cytoplasm.
  Both: camera offset 100 counts plus Poisson (shot) noise; pixel size written to the files: 0.2 um (a made-up
  scale so the scale bar has something to show). One file per channel, as the desktop app expects (T, Y, X) stacks.
- demo_mask.tif: 24 x 128 x 128 uint8 (0/255), Deflate-compressed. Made by the repository's own code: each cytoplasm
  frame is smoothed with ImageProcessor.apply_filter('gaussian', sigma=2) and segmented with
  ImageProcessor.segment_image(method='threshold') (Otsu's threshold).

Run: python scripts/make_demo_data.py [--out web/static/demo]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.image_processor import ImageProcessor  # noqa: E402

SEED = 7
T, H, W = 24, 128, 128
PIXEL_UM = 0.2
OFFSET = 100
CHANNELS = ("cytoplasm", "puncta")


def cells(rng):
    """Six non-overlapping ellipses: (cy, cx, ry, rx, angle, brightness)."""
    out = []
    while len(out) < 6:
        cy, cx = rng.uniform(18, H - 18), rng.uniform(18, W - 18)
        if all(np.hypot(cy - c[0], cx - c[1]) > 34 for c in out):
            out.append((cy, cx, rng.uniform(10, 15), rng.uniform(13, 19), rng.uniform(0, np.pi), rng.uniform(600, 1100)))
    return out


def ellipse_field(cell_list):
    yy, xx = np.mgrid[:H, :W].astype(float)
    field = np.zeros((H, W))
    for cy, cx, ry, rx, ang, b in cell_list:
        dy, dx = yy - cy, xx - cx
        u = (dx * np.cos(ang) + dy * np.sin(ang)) / rx
        v = (-dx * np.sin(ang) + dy * np.cos(ang)) / ry
        r = np.sqrt(u ** 2 + v ** 2)
        field += b / (1 + np.exp((r - 1) * 12))  # soft edge
    return field


def puncta(rng, cell_list):
    """Spot positions inside the cells: (y, x, base amplitude)."""
    spots = []
    for cy, cx, ry, rx, ang, _ in cell_list:
        for _ in range(4):
            a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.7)
            u, v = rr * np.cos(a) * rx, rr * np.sin(a) * ry
            spots.append((cy + u * np.sin(ang) + v * np.cos(ang), cx + u * np.cos(ang) - v * np.sin(ang),
                          rng.uniform(400, 900)))
    return spots


def make(out: Path):
    rng = np.random.default_rng(SEED)
    cl = cells(rng)
    base = ellipse_field(cl)
    sp = puncta(rng, cl)
    yy, xx = np.mgrid[:H, :W].astype(float)
    stack = np.zeros((T, 2, H, W), dtype=np.uint16)
    for t in range(T):
        c1 = base * 0.99 ** t
        c2 = np.zeros((H, W))
        for y, x, a in sp:
            c2 += a * rng.uniform(0.2, 1.0) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.3 ** 2))
        c2 += 0.08 * base  # dim cytoplasmic background in channel 2
        for c, img in enumerate((c1, c2)):
            stack[t, c] = np.clip(rng.poisson(img + OFFSET), 0, 65535)
    out.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(CHANNELS):
        tifffile.imwrite(out / f"demo_{name}.tif", stack[:, c], imagej=True, resolution=(1 / PIXEL_UM, 1 / PIXEL_UM),
                         metadata={"axes": "TYX", "unit": "um", "finterval": 1.0})
    proc = ImageProcessor()
    mask = np.zeros((T, H, W), dtype=np.uint8)
    for t in range(T):
        smooth = proc.apply_filter(stack[t, 0], "gaussian", sigma=2)
        mask[t] = proc.segment_image(smooth, method="threshold").astype(np.uint8) * 255
    tifffile.imwrite(out / "demo_mask.tif", mask, compression="zlib", metadata={"axes": "TYX"})
    return stack, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "web" / "static" / "demo"))
    stack, mask = make(Path(ap.parse_args().out))
    print(f"demo_cytoplasm.tif, demo_puncta.tif {stack[:, 0].shape} {stack.dtype}; demo_mask.tif {mask.shape}, "
          f"{(mask > 0).mean() * 100:.1f} % of pixels inside the mask")


if __name__ == "__main__":
    main()

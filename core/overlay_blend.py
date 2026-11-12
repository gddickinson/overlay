"""Mask-over-fluorescence compositing used by the image view (Qt-free, so the web app and tests can use it too).

blend_overlay(image, mask, alpha, color) returns an (H, W, 3) uint8 image: the fluorescence frame scaled to 8 bits
(min-max of the frame, as ImageStack.get_normalized_frame does) and, where mask > 0, mixed with `color` at opacity
`alpha`:  out = base * (1 - alpha * m) + color * alpha * m,  truncated to uint8.
"""
import numpy as np


def to_uint8(image):
    """Scale a frame to 0-255 by its own min and max (uint8 frames are returned unchanged)."""
    if image.dtype == np.uint8:
        return image
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi == lo:
        hi = lo + 1
    return ((image.astype(float) - lo) / (hi - lo) * 255).astype(np.uint8)


def blend_overlay(image, mask, alpha, color):
    """Composite a binary mask in `color` (RGB 0-255) over a 2D or RGB frame at opacity `alpha` (0-1)."""
    base = to_uint8(np.asarray(image))
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=2)
    m = (np.asarray(mask) > 0).astype(float)
    out = np.empty(base.shape, dtype=np.uint8)
    for i in range(3):
        out[..., i] = (base[..., i] * (1 - alpha * m) + color[i] * alpha * m).astype(np.uint8)
    return out

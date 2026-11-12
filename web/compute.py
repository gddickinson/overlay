"""Python-side processing for the local web app: frames arrive as base64 little-endian arrays, run through the
repository's ImageProcessor, and go back the same way. The static site runs the JS ports instead and shows a
"needs the local app" notice for operations that exist only here (Frangi, CLAHE, local threshold, blobs, corners,
registration).
"""
import base64

import numpy as np

from core.image_processor import ImageProcessor

DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"}
MAX_PIXELS = 64_000_000


def decode(arr):
    """{'shape': [...], 'dtype': 'uint16', 'data': base64} -> numpy array."""
    dtype = arr.get("dtype")
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    shape = [int(s) for s in arr["shape"]]
    if not shape or any(s < 1 for s in shape) or int(np.prod(shape)) > MAX_PIXELS:
        raise ValueError("bad or too large shape")
    raw = base64.b64decode(arr["data"])
    a = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))
    if a.size != int(np.prod(shape)):
        raise ValueError(f"{a.size} values for shape {shape}")
    return a.reshape(shape).astype(dtype)


def encode(a):
    a = np.asarray(a)
    if a.dtype == bool:
        a = a.astype(np.uint8)
    if str(a.dtype) not in DTYPES:
        a = a.astype(np.float64)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "data": base64.b64encode(np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes()).decode()}


def process(op, params, frame):
    """Run one ImageProcessor operation on a 2D frame. Returns {'kind': 'image'|'mask'|'points', ...}."""
    p, proc = dict(params or {}), ImageProcessor()
    if op in ("gaussian", "median", "bilateral", "sobel", "laplacian", "frangi"):
        if op == "median":
            p["size"] = int(p.get("size", 3))
        return {"kind": "image", "image": encode(proc.apply_filter(frame, op, **p))}
    if op in ("rescale", "equalize"):
        return {"kind": "image", "image": encode(proc.enhance_contrast(frame, op, **p))}
    if op == "adaptive_contrast":
        p["kernel_size"] = int(p.get("kernel_size", 64))
        return {"kind": "image", "image": encode(proc.enhance_contrast(frame, "adaptive", **p))}
    if op == "threshold":
        return {"kind": "mask", "image": encode(proc.segment_image(frame, "threshold"))}
    if op == "adaptive_threshold":
        p["block_size"] = int(p.get("block_size", 35)) | 1
        return {"kind": "mask", "image": encode(proc.segment_image(frame.astype(float), "adaptive", **p))}
    if op in ("blob", "corner"):
        pts = proc.detect_features(frame.astype(float), method=op, **p)
        return {"kind": "points", "points": np.asarray(pts, dtype=float).tolist()}
    raise ValueError(f"unknown operation {op!r}")


def register(frames, reference, params):
    """Phase-correlation registration of every frame to frames[reference] (as the desktop's 'self' registration)."""
    proc, out, shifts = ImageProcessor(), frames.copy(), []
    ref = frames[reference]
    for i in range(frames.shape[0]):
        if i == reference:
            shifts.append([0.0, 0.0])
            continue
        reg, shift = proc.register_images(ref, frames[i], "phase", **(params or {}))
        out[i] = reg
        shifts.append([float(s) for s in shift] if shift is not None else None)
    return {"frames": encode(out), "shifts": shifts}

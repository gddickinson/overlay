"""Arrange a tifffile series by its axes into (frames, channels, Y, X), the layout ImageStack works with.

Time and any other page axis (T, I, Q, ...) and Z are merged into frames, T-major with Z fastest; C is the channel
axis. The browser reader (web/static/tiff_reader.js) uses the same convention. Files without a C or Z axis are left
to ImageStack's original handling (plain stacks, RGB samples).
"""
import numpy as np


def needs_arranging(axes):
    """True when the series has a channel or z axis that tifffile.imread would leave in the wrong place."""
    return 'C' in axes or 'Z' in axes


def arrange(data, axes):
    """(frames, C, Y, X) array from `data` with tifffile axes such as 'TCYX', 'TZCYX', 'CYX' or 'ZCYX'."""
    if 'S' in axes:  # samples inside a channel stack: treat them as more channels
        axes = axes.replace('S', 'C') if 'C' not in axes else axes
    frame_axes = [i for i, a in enumerate(axes) if a not in 'CYXZS']
    frame_axes += [i for i, a in enumerate(axes) if a == 'Z']
    chan_axes = [i for i, a in enumerate(axes) if a in 'CS']
    order = frame_axes + chan_axes + [axes.index('Y'), axes.index('X')]
    out = np.transpose(data, order)
    n_frames = int(np.prod([data.shape[i] for i in frame_axes])) if frame_axes else 1
    n_chan = int(np.prod([data.shape[i] for i in chan_axes])) if chan_axes else 1
    return out.reshape(n_frames, n_chan, out.shape[-2], out.shape[-1])

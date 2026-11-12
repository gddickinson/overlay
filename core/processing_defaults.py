"""Default parameters of every ImageProcessor operation, in one place.

ImageProcessor (core/image_processor.py) falls back to these when a keyword is not passed, and the web interface
(web/ops.py) shows the same values as its initial settings, so the desktop and the browser cannot drift apart.
Keys: DEFAULTS[<group>][<operation>][<parameter>].
"""

DEFAULTS = {
    'filter': {
        'gaussian': {'sigma': 1.0},
        'median': {'size': 3},
        'bilateral': {'sigma_color': 0.1, 'sigma_spatial': 1},
        'frangi': {'scale_range': (1, 10), 'scale_step': 2},
    },
    'contrast': {
        'rescale': {'p_low': 2, 'p_high': 98},
        'adaptive': {'kernel_size': 64, 'clip_limit': 0.01},
    },
    'registration': {
        'phase': {'upsample_factor': 10},
    },
    'segment': {
        'threshold': {'threshold': None},  # None: Otsu's threshold
        'adaptive': {'block_size': 35, 'offset': 0},
        'watershed': {'low_marker': 30, 'high_marker': 150},
    },
    'features': {
        'blob': {'min_sigma': 1, 'max_sigma': 10, 'num_sigma': 5, 'threshold': 0.1},
        'corner': {'min_distance': 5, 'threshold': 0.1},
    },
}

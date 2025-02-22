"""
Configuration management for the application.
"""

import os
import json
from pathlib import Path
import logging

# Default configuration
DEFAULT_CONFIG = {
    "appearance": {
        "theme": "dark",
        "font_size": 10,
        "window_size": [1000, 800],
        "window_position": [100, 100],
        "maximize_on_start": False
    },
    "display": {
        "default_mask_color": [255, 0, 0],
        "default_overlay_alpha": 0.5,
        "auto_contrast": True,
        "colormap": "viridis",
        "show_scale_bar": True,
        "scale_bar_size": 50,
        "scale_bar_units": "μm",
        "pixels_per_unit": 1.0
    },
    "navigation": {
        "frame_rate": 10,
        "loop_playback": True,
        "bidirectional_playback": False,
        "mouse_wheel_sensitivity": 1.0
    },
    "processing": {
        "default_roi_color": [0, 255, 0],
        "default_roi_width": 2,
        "smoothing_kernel_size": 3,
        "registration_max_iterations": 100,
        "registration_precision": 0.1
    },
    "export": {
        "default_directory": "",
        "default_format": "png",
        "default_dpi": 300,
        "default_quality": 95,
        "include_timestamp": True,
        "include_analysis_info": True
    },
    "recent_files": {
        "fluorescence": [],
        "mask": []
    },
    "keyboard_shortcuts": {
        "next_frame": "Right",
        "previous_frame": "Left",
        "play_pause": "Space",
        "toggle_mask": "M",
        "toggle_fluorescence": "F",
        "zoom_in": "Ctrl++",
        "zoom_out": "Ctrl+-",
        "reset_view": "Ctrl+0",
        "save": "Ctrl+S",
        "open_fluorescence": "Ctrl+O",
        "open_mask": "Ctrl+Shift+O"
    }
}


def get_config_path(custom_path=None):
    """Get the path to the configuration file."""
    if custom_path:
        return Path(custom_path)
    
    if sys.platform == 'win32':
        config_dir = Path(os.path.expandvars('%APPDATA%')) / "tiff_stack_viewer"
    else:
        config_dir = Path(os.path.expanduser('~')) / ".tiff_stack_viewer"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config(custom_path=None):
    """Load configuration from file or return default if file doesn't exist."""
    logger = logging.getLogger('tiff_stack_viewer')
    config_path = get_config_path(custom_path)
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update default config with loaded values
            _recursive_update(config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Configuration file not found at {config_path}")
        logger.info("Using default configuration")
    
    return config


def save_config(config, custom_path=None):
    """Save configuration to file."""
    logger = logging.getLogger('tiff_stack_viewer')
    config_path = get_config_path(custom_path)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def _recursive_update(d, u):
    """Recursively update a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _recursive_update(d[k], v)
        else:
            d[k] = v


def reset_to_defaults(custom_path=None):
    """Reset configuration to defaults."""
    logger = logging.getLogger('tiff_stack_viewer')
    config_path = get_config_path(custom_path)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.info("Configuration reset to defaults")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        return DEFAULT_CONFIG.copy()


import sys  # Add this import at the top

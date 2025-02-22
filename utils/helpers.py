"""
Helper functions for the Advanced TIFF Stack Viewer.

This module contains various utility functions used across the application.
"""

import os
import sys
import platform
import uuid
import numpy as np


def get_application_dir():
    """Get the application data directory based on the platform."""
    if platform.system() == 'Windows':
        app_dir = os.path.join(os.environ['APPDATA'], 'TiffStackViewer')
    elif platform.system() == 'Darwin':  # macOS
        app_dir = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'TiffStackViewer')
    else:  # Linux and others
        app_dir = os.path.join(os.path.expanduser('~'), '.tiff_stack_viewer')
    
    # Create directory if it doesn't exist
    os.makedirs(app_dir, exist_ok=True)
    
    return app_dir


def generate_unique_id(prefix=''):
    """Generate a unique identifier."""
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def normalize_array(array, min_val=0, max_val=255):
    """Normalize array values to specified range."""
    if array.min() == array.max():
        return np.zeros_like(array)
    
    normalized = (array - array.min()) / (array.max() - array.min())
    return normalized * (max_val - min_val) + min_val


def ensure_directory(directory):
    """Ensure a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_file_extension(file_path):
    """Get the file extension from a path."""
    return os.path.splitext(file_path)[1].lower()


def get_file_size_str(file_path):
    """Get the file size as a human-readable string."""
    if not os.path.exists(file_path):
        return "N/A"
    
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def memory_usage_info():
    """Get current memory usage information."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
        'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
        'percent': process.memory_percent()
    }


def convert_colormap_to_lut(colormap_name):
    """Convert a matplotlib colormap name to a PyQtGraph lookup table."""
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(colormap_name)
    lut = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        color = cmap(i / 255.0)
        lut[i, 0] = int(color[0] * 255)
        lut[i, 1] = int(color[1] * 255)
        lut[i, 2] = int(color[2] * 255)
    
    return lut


def time_function(func):
    """Decorator to measure execution time of a function."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    
    return wrapper

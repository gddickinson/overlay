"""
Image loading and management functionality.
"""

import os
import logging
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
import h5py
from scipy.ndimage import zoom
from skimage import io, transform, exposure


class ImageStack:
    """Class for managing image stacks with metadata."""

    def __init__(self, logger=None):
        """Initialize empty image stack."""
        self.logger = logger or logging.getLogger('tiff_stack_viewer')
        self.data = None
        self.file_path = None
        self.metadata = {}
        self.dimensions = None
        self.current_frame = 0
        self.max_frames = 0

    def load_tiff(self, file_path):
        """Load a TIFF stack from file."""
        self.logger.info(f"Loading TIFF stack from {file_path}")

        try:
            # Load the TIFF file
            self.data = imread(file_path)
            self.file_path = file_path

            # Reshape if it's a single frame
            if len(self.data.shape) == 2:
                self.data = self.data.reshape(1, *self.data.shape)
                self.logger.debug("Reshaped single frame to add dimension")

            # Update dimensions and metadata
            self._update_dimensions()
            self._extract_metadata()

            self.logger.info(f"Loaded TIFF stack with shape {self.data.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading TIFF stack: {e}")
            return False

    def _update_dimensions(self):
        """Update dimension information based on loaded data."""
        if self.data is not None:
            self.dimensions = {
                'frames': self.data.shape[0],
                'height': self.data.shape[1],
                'width': self.data.shape[2],
                'channels': self.data.shape[3] if len(self.data.shape) > 3 else 1
            }
            self.max_frames = self.dimensions['frames']
        else:
            self.dimensions = None
            self.max_frames = 0

    def _extract_metadata(self):
        """Extract metadata from TIFF file if available."""
        # Basic metadata
        self.metadata = {
            'filename': os.path.basename(self.file_path),
            'directory': os.path.dirname(self.file_path),
            'size_mb': os.path.getsize(self.file_path) / (1024 * 1024),
            'dtype': str(self.data.dtype),
            'dimensions': self.dimensions
        }

        # Try to extract more metadata from TIFF tags (if available)
        try:
            from tifffile import TiffFile
            with TiffFile(self.file_path) as tif:
                for tag in tif.pages[0].tags.values():
                    self.metadata[tag.name] = tag.value

                # Check for ImageJ metadata
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    for key, value in tif.imagej_metadata.items():
                        self.metadata[f'ImageJ_{key}'] = value
        except Exception as e:
            self.logger.debug(f"Could not extract extended metadata: {e}")

    def get_frame(self, index=None):
        """Get a specific frame from the stack."""
        if self.data is None:
            return None

        if index is None:
            index = self.current_frame

        # Ensure index is in valid range
        index = max(0, min(index, self.max_frames - 1))
        self.current_frame = index

        return self.data[index]

    def get_normalized_frame(self, index=None, min_max=None):
        """Get a frame normalized to 0-255 range for display."""
        frame = self.get_frame(index)
        if frame is None:
            return None

        # If min_max not provided, calculate from the frame
        if min_max is None:
            min_val = frame.min()
            max_val = frame.max()
        else:
            min_val, max_val = min_max

        # Avoid division by zero
        if max_val == min_val:
            max_val = min_val + 1

        # Normalize to 0-255 range
        normalized = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized

    def save_tiff(self, file_path, frames=None):
        """Save stack or specific frames to a TIFF file."""
        if self.data is None:
            self.logger.error("No data to save")
            return False

        try:
            if frames is None:
                # Save the entire stack
                imwrite(file_path, self.data, metadata=self.metadata)
            else:
                # Save specific frames
                if isinstance(frames, int):
                    frames = [frames]

                # Extract the frames
                frames_data = self.data[frames]
                imwrite(file_path, frames_data, metadata=self.metadata)

            self.logger.info(f"Saved TIFF stack to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving TIFF stack: {e}")
            return False

    def create_z_projection(self, method='max'):
        """Create a Z projection of the stack using the specified method."""
        if self.data is None:
            return None

        self.logger.info(f"Creating {method} Z-projection")

        try:
            if method == 'max':
                return np.max(self.data, axis=0)
            elif method == 'min':
                return np.min(self.data, axis=0)
            elif method == 'mean':
                return np.mean(self.data, axis=0)
            elif method == 'median':
                return np.median(self.data, axis=0)
            elif method == 'std':
                return np.std(self.data, axis=0)
            else:
                self.logger.error(f"Unknown projection method: {method}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating Z-projection: {e}")
            return None

    def apply_contrast(self, p_low=2, p_high=98):
        """Apply contrast stretching to the entire stack."""
        if self.data is None:
            return False

        try:
            # Calculate percentiles from the entire stack
            low = np.percentile(self.data, p_low)
            high = np.percentile(self.data, p_high)

            # Apply contrast stretching
            self.data = exposure.rescale_intensity(
                self.data,
                in_range=(low, high),
                out_range=self.data.dtype.type
            )

            self.logger.info(f"Applied contrast stretching ({p_low}% - {p_high}%)")
            return True
        except Exception as e:
            self.logger.error(f"Error applying contrast: {e}")
            return False

    def resize(self, scale_factor):
        """Resize the stack by the given scale factor."""
        if self.data is None or scale_factor == 1.0:
            return False

        try:
            # Calculate new dimensions
            new_shape = (
                self.data.shape[0],  # Keep number of frames
                int(self.data.shape[1] * scale_factor),  # Height
                int(self.data.shape[2] * scale_factor)   # Width
            )

            # Add channel dimension if present
            if len(self.data.shape) > 3:
                new_shape = new_shape + (self.data.shape[3],)

            # Resize using scipy.ndimage zoom
            zoom_factors = (1,) + (scale_factor,) * 2
            if len(self.data.shape) > 3:
                zoom_factors = zoom_factors + (1,)

            self.data = zoom(self.data, zoom_factors, order=1)

            # Update dimensions
            self._update_dimensions()

            self.logger.info(f"Resized stack by factor {scale_factor}")
            return True
        except Exception as e:
            self.logger.error(f"Error resizing stack: {e}")
            return False


class ImageLoader:
    """Class to handle loading various image formats."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('tiff_stack_viewer')
        self.supported_formats = {
            'tiff': ['.tif', '.tiff'],
            'hdf5': ['.h5', '.hdf5'],
            'image_series': ['.png', '.jpg', '.jpeg', '.bmp']
        }

    def load_file(self, file_path):
        """Load image data from file and return an ImageStack."""
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        # Check file extension
        ext = file_path.suffix.lower()

        # Create a new ImageStack
        image_stack = ImageStack(self.logger)

        # Load based on file type
        if ext in self.supported_formats['tiff']:
            # Load TIFF file
            success = image_stack.load_tiff(str(file_path))

        elif ext in self.supported_formats['hdf5']:
            # Load HDF5 file
            success = self._load_hdf5(image_stack, file_path)

        elif ext in self.supported_formats['image_series']:
            # Check if it's part of a series
            success = self._load_image_series(image_stack, file_path)

        else:
            self.logger.error(f"Unsupported file format: {ext}")
            return None

        if success:
            return image_stack
        return None

    def _load_hdf5(self, image_stack, file_path):
        """Load data from HDF5 file into ImageStack."""
        try:
            with h5py.File(file_path, 'r') as f:
                # List available datasets
                datasets = list(f.keys())
                self.logger.debug(f"HDF5 datasets: {datasets}")

                if not datasets:
                    self.logger.error("No datasets found in HDF5 file")
                    return False

                # Use the first dataset or 'data' if available
                dataset_name = 'data' if 'data' in datasets else datasets[0]
                data = f[dataset_name][:]

                # Ensure data has the right dimensions
                if len(data.shape) == 2:
                    # Single frame
                    data = data.reshape(1, *data.shape)
                elif len(data.shape) == 3 and data.shape[2] <= 4:
                    # Could be a single RGB/RGBA frame
                    # Check if third dimension is color channels
                    if data.shape[2] <= 4:
                        data = data.reshape(1, *data.shape)

                # Store data and file path
                image_stack.data = data
                image_stack.file_path = str(file_path)

                # Update dimensions and metadata
                image_stack._update_dimensions()

                # Extract metadata from HDF5 attributes
                metadata = {}
                for key, value in f.attrs.items():
                    metadata[key] = value

                image_stack.metadata = {
                    'filename': file_path.name,
                    'directory': str(file_path.parent),
                    'size_mb': os.path.getsize(file_path) / (1024 * 1024),
                    'dtype': str(data.dtype),
                    'dimensions': image_stack.dimensions,
                    'hdf5_datasets': datasets,
                    'hdf5_attributes': metadata
                }
            return True

        except Exception as e:
            self.logger.error(f"Error loading HDF5 file: {e}")
            return False

    def _load_image_series(self, image_stack, file_path):
        """Load a series of image files as a stack."""
        try:
            # Get directory and filename pattern
            directory = file_path.parent
            filename = file_path.name

            # Find files with similar names
            base_name = filename.split('.')[0]
            extension = file_path.suffix

            # Look for numbered sequence like name_001.png, name_002.png, etc.
            import re
            pattern = re.compile(rf"{re.escape(base_name)}_?(\d+){re.escape(extension)}")

            matching_files = []
            for f in directory.glob(f"*{extension}"):
                match = pattern.match(f.name)
                if match:
                    index = int(match.group(1))
                    matching_files.append((index, f))

            # If no numbered sequence found, try alphabetical order
            if not matching_files:
                pattern = re.compile(rf"{re.escape(base_name)}.*{re.escape(extension)}")
                matching_files = [(i, f) for i, f in enumerate(directory.glob(f"{base_name}*{extension}"))]

            # Sort by index
            matching_files.sort(key=lambda x: x[0])

            if not matching_files:
                # No series found, load single image
                self.logger.info("No image series found, loading single image")
                img = io.imread(str(file_path))
                if len(img.shape) == 2:
                    # Grayscale image
                    img = img.reshape(1, *img.shape)
                elif len(img.shape) == 3 and img.shape[2] <= 4:
                    # RGB/RGBA image
                    img = img.reshape(1, *img.shape)

                image_stack.data = img
                image_stack.file_path = str(file_path)
            else:
                # Load all images in the series
                self.logger.info(f"Loading image series with {len(matching_files)} files")

                # Load first image to get dimensions
                first_img = io.imread(str(matching_files[0][1]))

                # Create array for all images
                if len(first_img.shape) == 2:
                    # Grayscale images
                    all_images = np.zeros((len(matching_files), *first_img.shape), dtype=first_img.dtype)
                else:
                    # Color images
                    all_images = np.zeros((len(matching_files), *first_img.shape), dtype=first_img.dtype)

                # Load all images
                all_images[0] = first_img
                for i, (_, img_path) in enumerate(matching_files[1:], 1):
                    img = io.imread(str(img_path))
                    # Check if dimensions match
                    if img.shape != first_img.shape:
                        img = transform.resize(img, first_img.shape, preserve_range=True).astype(first_img.dtype)
                    all_images[i] = img

                image_stack.data = all_images
                image_stack.file_path = str(directory / f"{base_name}_series")

            # Update dimensions and metadata
            image_stack._update_dimensions()
            image_stack._extract_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Error loading image series: {e}")
            return False

    def get_supported_formats_filter(self):
        """Return a file dialog filter string for supported formats."""
        filters = []
        # TIFF files
        tiff_exts = " ".join(f"*{ext}" for ext in self.supported_formats['tiff'])
        filters.append(f"TIFF Files ({tiff_exts})")

        # HDF5 files
        hdf5_exts = " ".join(f"*{ext}" for ext in self.supported_formats['hdf5'])
        filters.append(f"HDF5 Files ({hdf5_exts})")

        # Image files
        img_exts = " ".join(f"*{ext}" for ext in self.supported_formats['image_series'])
        filters.append(f"Image Files ({img_exts})")

        # All supported formats
        all_exts = " ".join(f"*{ext}" for formats in self.supported_formats.values() for ext in formats)
        filters.insert(0, f"All Supported Formats ({all_exts})")

        return ";;".join(filters)

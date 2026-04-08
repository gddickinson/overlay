# Advanced TIFF Stack Viewer

A PyQt6-based desktop application for viewing, analyzing, and processing microscopy TIFF stacks. Designed for fluorescence microscopy workflows where a fluorescence image stack needs to be overlaid with a binary mask for region-specific analysis.

## Features

### Visualization
- **Multi-stack overlay**: Load fluorescence and mask TIFF stacks simultaneously with controllable opacity and colormap
- **Frame navigation**: Browse through time-series frames using a timeline widget, navigation bar, or keyboard shortcuts
- **Histogram display**: Real-time intensity histogram with adjustable contrast/brightness
- **Z-projection**: Maximum, minimum, and average intensity projections across the stack

### Analysis
- **ROI tools**: Draw and manage regions of interest for targeted measurements
- **Time-series analysis**: Plot intensity changes over frames within selected ROIs
- **Basic statistics**: Per-frame mean, median, standard deviation, min, max, and SNR
- **Advanced analysis**: PCA and clustering via scikit-learn integration
- **Registration**: Align misaligned image stacks
- **Measurement tools**: Distance, angle, and area measurements on images

### Processing
- **Image processing**: Filters, contrast adjustment, and basic image math
- **Export**: Save visualizations as images, movies, or tabular data files

### Command-Line Interface
- Load files directly from the command line:
  ```bash
  python main.py --fluorescence path/to/fluor.tif --mask path/to/mask.tif --debug
  ```

## Project Structure

```
overlay/
├── main.py                     # Entry point with CLI argument parsing
├── requirements.txt            # Dependencies
├── core/
│   ├── image_loader.py         # TIFF loading and ImageStack management
│   ├── image_processor.py      # Image processing algorithms
│   ├── data_analyzer.py        # Statistics, PCA, clustering analysis
│   └── export_manager.py       # Export to images, video, and data files
├── gui/
│   ├── main_window.py          # Main application window (signals, menus, layout)
│   ├── image_view.py           # Enhanced pyqtgraph-based image viewer
│   ├── controls_panel.py       # Display and overlay control panels
│   ├── dialogs.py              # Settings, contrast, projection, registration,
│   │                           #   analysis, export, and results dialogs
│   └── widgets/
│       ├── histogram_widget.py # Live intensity histogram
│       ├── roi_tools.py        # ROI drawing and management toolbar
│       ├── navigation_bar.py   # Frame navigation controls
│       └── timeline_widget.py  # Timeline scrubber widget
├── utils/
│   ├── config.py               # Application configuration (load/save)
│   ├── logger.py               # Logging setup
│   ├── helpers.py              # Utility functions
│   └── tiff_stack_viewer.py    # Standalone viewer utility
├── standalone/                 # Standalone tool scripts
└── example_data/               # Sample TIFF stacks for testing
```

## Requirements

- Python >= 3.8
- PyQt6 >= 6.2
- pyqtgraph >= 0.12
- NumPy >= 1.20
- tifffile >= 2021.8
- scikit-image >= 0.18
- SciPy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4
- h5py >= 3.6
- OpenCV (cv2)
- scikit-learn

## Installation

```bash
git clone <repository-url>
cd overlay
pip install -r requirements.txt
```

## Usage

Launch the application:

```bash
python main.py
```

Or load files directly:

```bash
python main.py -f fluorescence_stack.tif -m mask_stack.tif
```

### Quick Start

1. **File > Open Fluorescence Stack** to load your fluorescence data
2. **File > Open Mask Stack** to load a binary segmentation mask
3. Use the timeline slider or arrow keys to navigate frames
4. Adjust overlay opacity and colormap in the Display panel
5. Draw ROIs to extract region-specific intensity traces
6. Use **Analysis** dialogs for statistics, PCA, or clustering
7. Export results via the **Export** menu

## Debugging

- Logs are saved to `~/.tiff_stack_viewer/logs/`
- Pass `--debug` on the command line for verbose logging
- Use **View > Show Log** within the application

## GUI Framework

Built with **PyQt6** and **pyqtgraph** for high-performance image rendering.

## License

MIT License -- see the LICENSE file for details.

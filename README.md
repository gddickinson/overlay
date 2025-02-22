# Advanced TIFF Stack Viewer

An advanced PyQt6-based application for viewing, analyzing, and processing microscopy TIFF stacks.

## Features

- **Multi-Stack Visualization**: Load and visualize multiple TIFF stacks simultaneously
- **Overlay Management**: Control visibility, opacity, and color of different layers
- **Navigation Tools**: Browse through frames with keyboard shortcuts and timeline
- **Image Processing**: Apply filters, adjust contrast/brightness, and perform basic image processing
- **ROI Analysis**: Define regions of interest and extract quantitative data
- **Time Series Analysis**: Plot intensity changes over frames
- **Z-Projection**: Create maximum, minimum, or average intensity projections
- **Export Options**: Save as images, movies, or data files
- **Registration Tools**: Align misaligned image stacks
- **3D Visualization**: Orthogonal views for Z-stacks
- **Measurement Tools**: Distance, angle, and area measurements

## Installation

1. Clone this repository:
```
git clone https://github.com/username/tiff_stack_viewer.git
cd tiff_stack_viewer
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Launch the application:
```
python main.py
```

### Quick Start Guide

1. Click "File" → "Open Fluorescence Stack" to load your fluorescence data
2. Click "File" → "Open Mask Stack" to load your binary mask
3. Use the timeline or slider to navigate through frames
4. Adjust overlay settings in the "Display" panel
5. Use ROI tools to select regions of interest
6. Export visualizations or data using the "Export" menu

## Development

The project is organized in a modular structure:
- `core/`: Core functionality for data handling and processing
- `gui/`: User interface components
- `utils/`: Helper functions and utilities

## Debugging

The application includes comprehensive logging functionality:
- Logs are saved to `~/.tiff_stack_viewer/logs/`
- Set the environment variable `TSV_DEBUG=1` to enable debug-level logging
- Use the "View" → "Show Log" menu option to view logs within the application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

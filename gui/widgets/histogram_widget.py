"""
Histogram widget for displaying image intensity distributions.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QSlider, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg


class HistogramWidget(QWidget):
    """Widget for displaying and manipulating image histograms."""

    # Signals
    levels_changed = pyqtSignal(tuple)  # min, max levels

    def __init__(self, parent=None):
        """Initialize histogram widget."""
        super().__init__(parent)

        # Initialize variables
        self.current_image = None
        self.auto_levels = True
        self.min_level = 0
        self.max_level = 255

        # Set up UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create histogram plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(False)

        # Create a PlotDataItem for the histogram
        self.histogram_plot = self.plot_widget.plot([], [],
                                                  fillLevel=0,
                                                  brush=(100,100,255,150))

        # Add plot to layout
        layout.addWidget(self.plot_widget)

        # Add controls
        controls_layout = QHBoxLayout()

        # Auto level checkbox
        self.auto_level_checkbox = QCheckBox("Auto Levels")
        self.auto_level_checkbox.setChecked(self.auto_levels)
        self.auto_level_checkbox.stateChanged.connect(self.on_auto_level_changed)
        controls_layout.addWidget(self.auto_level_checkbox)

        # Min level slider
        min_level_layout = QHBoxLayout()
        min_level_layout.addWidget(QLabel("Min:"))
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 255)
        self.min_slider.setValue(self.min_level)
        self.min_slider.valueChanged.connect(self.on_min_level_changed)
        min_level_layout.addWidget(self.min_slider)
        self.min_value_label = QLabel(f"{self.min_level}")
        min_level_layout.addWidget(self.min_value_label)

        # Max level slider
        max_level_layout = QHBoxLayout()
        max_level_layout.addWidget(QLabel("Max:"))
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 255)
        self.max_slider.setValue(self.max_level)
        self.max_slider.valueChanged.connect(self.on_max_level_changed)
        max_level_layout.addWidget(self.max_slider)
        self.max_value_label = QLabel(f"{self.max_level}")
        max_level_layout.addWidget(self.max_value_label)

        # Add sliders to controls
        controls_layout.addLayout(min_level_layout)
        controls_layout.addLayout(max_level_layout)

        layout.addLayout(controls_layout)

    def update_histogram(self, data=None):
        """Update histogram with new image data.

        Args:
            data: Either an ImageStack object or a numpy array
        """
        if data is None:
            return

        # Extract image data whether it's an ImageStack or numpy array
        if hasattr(data, 'get_frame'):
            # It's an ImageStack
            image = data.get_frame(data.current_frame)
        else:
            # It's a numpy array
            image = data

        if image is None:
            return

        self.current_image = image

        # Compute histogram
        hist, bins = np.histogram(image.flatten(), bins=256)

        # Update histogram plot
        self.histogram_plot.setData(bins[:-1], hist)

        if self.auto_levels:
            # Calculate auto levels (1% and 99% percentiles)
            flat_image = image.flatten()
            min_level = np.percentile(flat_image, 1)
            max_level = np.percentile(flat_image, 99)

            # Update sliders
            self.min_slider.setValue(int(min_level))
            self.max_slider.setValue(int(max_level))

            # Emit signal
            self.levels_changed.emit((min_level, max_level))
        else:
            # Use manual levels
            self.levels_changed.emit((self.min_level, self.max_level))

    def on_levels_changed(self):
        """Handle changes to histogram levels."""
        if not self.auto_levels:
            min_level = self.min_level
            max_level = self.max_level

            # Update slider values
            self.min_slider.setValue(int(min_level))
            self.max_slider.setValue(int(max_level))

            # Update labels
            self.min_value_label.setText(f"{int(min_level)}")
            self.max_value_label.setText(f"{int(max_level)}")

            # Emit signal
            self.levels_changed.emit((min_level, max_level))

    def on_auto_level_changed(self, state):
        """Handle changes to auto-level checkbox."""
        self.auto_levels = (state == Qt.CheckState.Checked)

        # Enable/disable sliders based on auto-level state
        self.min_slider.setEnabled(not self.auto_levels)
        self.max_slider.setEnabled(not self.auto_levels)

        # Update histogram if we have an image
        if self.current_image is not None:
            self.update_histogram(self.current_image)

    def on_min_level_changed(self, value):
        """Handle changes to min level slider."""
        self.min_level = value
        self.min_value_label.setText(f"{value}")

        # Ensure min level is always less than max level
        if self.min_level >= self.max_level:
            self.max_level = self.min_level + 1
            self.max_slider.setValue(self.max_level)

        # Emit signal for level changes
        if not self.auto_levels:
            self.levels_changed.emit((self.min_level, self.max_level))

    def on_max_level_changed(self, value):
        """Handle changes to max level slider."""
        self.max_level = value
        self.max_value_label.setText(f"{value}")

        # Ensure max level is always greater than min level
        if self.max_level <= self.min_level:
            self.min_level = self.max_level - 1
            self.min_slider.setValue(self.min_level)

        # Emit signal for level changes
        if not self.auto_levels:
            self.levels_changed.emit((self.min_level, self.max_level))

    def get_levels(self):
        """Get current histogram levels."""
        return (self.min_level, self.max_level)

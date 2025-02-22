"""
Dialog windows for the Advanced TIFF Stack Viewer.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QLabel, QLineEdit, QPushButton, QComboBox,
                            QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog,
                            QTabWidget, QGroupBox, QSlider, QTextEdit,
                            QDialogButtonBox, QListWidget, QRadioButton,
                            QScrollArea, QTableWidget, QTableWidgetItem, QWidget,
                            QMessageBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
import pandas as pd
from scipy import stats

class SettingsDialog(QDialog):
    """Dialog for editing application settings."""

    def __init__(self, config, parent=None):
        """Initialize settings dialog."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')
        self.config = config.copy()

        # Setup UI
        self.setWindowTitle("Settings")
        self.resize(500, 400)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Add tabs
        self.appearance_tab = self.create_appearance_tab()
        self.tab_widget.addTab(self.appearance_tab, "Appearance")

        self.display_tab = self.create_display_tab()
        self.tab_widget.addTab(self.display_tab, "Display")

        self.navigation_tab = self.create_navigation_tab()
        self.tab_widget.addTab(self.navigation_tab, "Navigation")

        self.processing_tab = self.create_processing_tab()
        self.tab_widget.addTab(self.processing_tab, "Processing")

        self.export_tab = self.create_export_tab()
        self.tab_widget.addTab(self.export_tab, "Export")

        self.shortcuts_tab = self.create_shortcuts_tab()
        self.tab_widget.addTab(self.shortcuts_tab, "Shortcuts")

        layout.addWidget(self.tab_widget)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                    QDialogButtonBox.StandardButton.Cancel |
                                    QDialogButtonBox.StandardButton.Apply |
                                    QDialogButtonBox.StandardButton.Reset)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_settings)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset_settings)

        layout.addWidget(button_box)

    def create_appearance_tab(self):
        """Create appearance settings tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Theme selection
        layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.setCurrentText(self.config['appearance']['theme'].capitalize())
        layout.addWidget(self.theme_combo, 0, 1)

        # Font size
        layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(self.config['appearance']['font_size'])
        layout.addWidget(self.font_size_spin, 1, 1)

        # Maximize on start
        self.maximize_checkbox = QCheckBox("Maximize window on startup")
        self.maximize_checkbox.setChecked(self.config['appearance']['maximize_on_start'])
        layout.addWidget(self.maximize_checkbox, 2, 0, 1, 2)

        layout.setRowStretch(3, 1)
        return tab

    def create_display_tab(self):
        """Create display settings tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Default mask color
        layout.addWidget(QLabel("Default Mask Color:"), 0, 0)
        self.mask_color_combo = QComboBox()
        self.mask_color_combo.addItems(["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"])

        # Set current color based on RGB values
        color_map = {
            (255, 0, 0): 0,      # Red
            (0, 255, 0): 1,      # Green
            (0, 0, 255): 2,      # Blue
            (255, 255, 0): 3,    # Yellow
            (255, 0, 255): 4,    # Magenta
            (0, 255, 255): 5     # Cyan
        }
        color_tuple = tuple(self.config['display']['default_mask_color'])
        self.mask_color_combo.setCurrentIndex(color_map.get(color_tuple, 0))

        layout.addWidget(self.mask_color_combo, 0, 1)

        # Default overlay alpha
        layout.addWidget(QLabel("Default Overlay Opacity:"), 1, 0)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.config['display']['default_overlay_alpha'] * 100))
        layout.addWidget(self.alpha_slider, 1, 1)
        self.alpha_value_label = QLabel(f"{self.config['display']['default_overlay_alpha']:.2f}")
        layout.addWidget(self.alpha_value_label, 1, 2)
        self.alpha_slider.valueChanged.connect(
            lambda value: self.alpha_value_label.setText(f"{value/100:.2f}")
        )

        # Auto contrast
        self.auto_contrast_checkbox = QCheckBox("Auto Contrast")
        self.auto_contrast_checkbox.setChecked(self.config['display']['auto_contrast'])
        layout.addWidget(self.auto_contrast_checkbox, 2, 0, 1, 2)

        # Colormap
        layout.addWidget(QLabel("Default Colormap:"), 3, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis",
                                     "gray", "hot", "cool", "jet", "rainbow"])
        self.colormap_combo.setCurrentText(self.config['display']['colormap'])
        layout.addWidget(self.colormap_combo, 3, 1)

        # Scale bar
        self.scale_bar_checkbox = QCheckBox("Show Scale Bar")
        self.scale_bar_checkbox.setChecked(self.config['display']['show_scale_bar'])
        layout.addWidget(self.scale_bar_checkbox, 4, 0, 1, 2)

        # Scale bar settings
        scale_bar_group = QGroupBox("Scale Bar Settings")
        scale_bar_layout = QGridLayout()

        scale_bar_layout.addWidget(QLabel("Size:"), 0, 0)
        self.scale_bar_size_spin = QSpinBox()
        self.scale_bar_size_spin.setRange(10, 1000)
        self.scale_bar_size_spin.setValue(self.config['display']['scale_bar_size'])
        scale_bar_layout.addWidget(self.scale_bar_size_spin, 0, 1)

        scale_bar_layout.addWidget(QLabel("Units:"), 1, 0)
        self.scale_bar_units_edit = QLineEdit(self.config['display']['scale_bar_units'])
        scale_bar_layout.addWidget(self.scale_bar_units_edit, 1, 1)

        scale_bar_layout.addWidget(QLabel("Pixels per Unit:"), 2, 0)
        self.pixels_per_unit_spin = QDoubleSpinBox()
        self.pixels_per_unit_spin.setRange(0.01, 1000.0)
        self.pixels_per_unit_spin.setValue(self.config['display']['pixels_per_unit'])
        scale_bar_layout.addWidget(self.pixels_per_unit_spin, 2, 1)

        scale_bar_group.setLayout(scale_bar_layout)
        layout.addWidget(scale_bar_group, 5, 0, 1, 3)

        layout.setRowStretch(6, 1)
        return tab

    def create_navigation_tab(self):
        """Create navigation settings tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Frame rate
        layout.addWidget(QLabel("Playback Frame Rate:"), 0, 0)
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setRange(1, 60)
        self.frame_rate_spin.setValue(self.config['navigation']['frame_rate'])
        layout.addWidget(self.frame_rate_spin, 0, 1)

        # Loop playback
        self.loop_checkbox = QCheckBox("Loop Playback")
        self.loop_checkbox.setChecked(self.config['navigation']['loop_playback'])
        layout.addWidget(self.loop_checkbox, 1, 0, 1, 2)

        # Bidirectional playback
        self.bidirectional_checkbox = QCheckBox("Bidirectional Playback")
        self.bidirectional_checkbox.setChecked(self.config['navigation']['bidirectional_playback'])
        layout.addWidget(self.bidirectional_checkbox, 2, 0, 1, 2)

        # Mouse wheel sensitivity
        layout.addWidget(QLabel("Mouse Wheel Sensitivity:"), 3, 0)
        self.wheel_sensitivity_spin = QDoubleSpinBox()
        self.wheel_sensitivity_spin.setRange(0.1, 5.0)
        self.wheel_sensitivity_spin.setSingleStep(0.1)
        self.wheel_sensitivity_spin.setValue(self.config['navigation']['mouse_wheel_sensitivity'])
        layout.addWidget(self.wheel_sensitivity_spin, 3, 1)
        layout.setRowStretch(4, 1)
        return tab

    def create_processing_tab(self):
        """Create processing settings tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Default ROI color
        layout.addWidget(QLabel("Default ROI Color:"), 0, 0)
        self.roi_color_combo = QComboBox()
        self.roi_color_combo.addItems(["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"])

        # Set current color based on RGB values
        color_map = {
            (255, 0, 0): 0,      # Red
            (0, 255, 0): 1,      # Green
            (0, 0, 255): 2,      # Blue
            (255, 255, 0): 3,    # Yellow
            (255, 0, 255): 4,    # Magenta
            (0, 255, 255): 5     # Cyan
        }
        color_tuple = tuple(self.config['processing']['default_roi_color'])
        self.roi_color_combo.setCurrentIndex(color_map.get(color_tuple, 1))  # Default to green

        layout.addWidget(self.roi_color_combo, 0, 1)

        # Default ROI width
        layout.addWidget(QLabel("Default ROI Width:"), 1, 0)
        self.roi_width_spin = QSpinBox()
        self.roi_width_spin.setRange(1, 10)
        self.roi_width_spin.setValue(self.config['processing']['default_roi_width'])
        layout.addWidget(self.roi_width_spin, 1, 1)

        # Smoothing kernel size
        layout.addWidget(QLabel("Smoothing Kernel Size:"), 2, 0)
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(1, 11)
        self.kernel_size_spin.setSingleStep(2)  # Only allow odd values
        self.kernel_size_spin.setValue(self.config['processing']['smoothing_kernel_size'])
        layout.addWidget(self.kernel_size_spin, 2, 1)

        # Registration settings
        registration_group = QGroupBox("Registration Settings")
        registration_layout = QGridLayout()

        registration_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(10, 1000)
        self.max_iterations_spin.setValue(self.config['processing']['registration_max_iterations'])
        registration_layout.addWidget(self.max_iterations_spin, 0, 1)

        registration_layout.addWidget(QLabel("Precision:"), 1, 0)
        self.precision_spin = QDoubleSpinBox()
        self.precision_spin.setRange(0.01, 1.0)
        self.precision_spin.setSingleStep(0.01)
        self.precision_spin.setValue(self.config['processing']['registration_precision'])
        registration_layout.addWidget(self.precision_spin, 1, 1)

        registration_group.setLayout(registration_layout)
        layout.addWidget(registration_group, 3, 0, 1, 2)

        layout.setRowStretch(4, 1)
        return tab

    def create_export_tab(self):
        """Create export settings tab."""
        tab = QWidget()
        layout = QGridLayout(tab)

        # Default directory
        layout.addWidget(QLabel("Default Export Directory:"), 0, 0)
        self.export_dir_edit = QLineEdit(self.config['export']['default_directory'])
        layout.addWidget(self.export_dir_edit, 0, 1)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_export_directory)
        layout.addWidget(browse_button, 0, 2)

        # Default format
        layout.addWidget(QLabel("Default Format:"), 1, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "tiff"])
        self.format_combo.setCurrentText(self.config['export']['default_format'])
        layout.addWidget(self.format_combo, 1, 1, 1, 2)

        # Default DPI
        layout.addWidget(QLabel("Default DPI:"), 2, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(self.config['export']['default_dpi'])
        layout.addWidget(self.dpi_spin, 2, 1, 1, 2)

        # Default quality
        layout.addWidget(QLabel("Default Quality (JPEG):"), 3, 0)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(self.config['export']['default_quality'])
        layout.addWidget(self.quality_spin, 3, 1, 1, 2)

        # Export options
        self.timestamp_checkbox = QCheckBox("Include Timestamp in Filenames")
        self.timestamp_checkbox.setChecked(self.config['export']['include_timestamp'])
        layout.addWidget(self.timestamp_checkbox, 4, 0, 1, 3)

        self.analysis_info_checkbox = QCheckBox("Include Analysis Information")
        self.analysis_info_checkbox.setChecked(self.config['export']['include_analysis_info'])
        layout.addWidget(self.analysis_info_checkbox, 5, 0, 1, 3)

        layout.setRowStretch(6, 1)
        return tab

    def create_shortcuts_tab(self):
        """Create keyboard shortcuts tab."""
        tab = QScrollArea()
        tab.setWidgetResizable(True)

        content_widget = QWidget()
        layout = QGridLayout(content_widget)

        # Add headers
        layout.addWidget(QLabel("<b>Action</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Shortcut</b>"), 0, 1)

        # Add all shortcuts
        row = 1
        for action, shortcut in self.config['keyboard_shortcuts'].items():
            # Convert action name to display name
            display_name = ' '.join(word.capitalize() for word in action.split('_'))

            layout.addWidget(QLabel(display_name), row, 0)

            # Create shortcut editor
            shortcut_edit = QLineEdit(shortcut)
            shortcut_edit.setObjectName(f"shortcut_{action}")
            layout.addWidget(shortcut_edit, row, 1)

            row += 1

        tab.setWidget(content_widget)
        return tab

    def browse_export_directory(self):
        """Browse for export directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", self.export_dir_edit.text()
        )

        if directory:
            self.export_dir_edit.setText(directory)

    def apply_settings(self):
        """Apply settings changes."""
        # Update appearance settings
        self.config['appearance']['theme'] = self.theme_combo.currentText().lower()
        self.config['appearance']['font_size'] = self.font_size_spin.value()
        self.config['appearance']['maximize_on_start'] = self.maximize_checkbox.isChecked()

        # Update display settings
        color_map = {
            0: [255, 0, 0],      # Red
            1: [0, 255, 0],      # Green
            2: [0, 0, 255],      # Blue
            3: [255, 255, 0],    # Yellow
            4: [255, 0, 255],    # Magenta
            5: [0, 255, 255]     # Cyan
        }
        self.config['display']['default_mask_color'] = color_map[self.mask_color_combo.currentIndex()]
        self.config['display']['default_overlay_alpha'] = self.alpha_slider.value() / 100.0
        self.config['display']['auto_contrast'] = self.auto_contrast_checkbox.isChecked()
        self.config['display']['colormap'] = self.colormap_combo.currentText()
        self.config['display']['show_scale_bar'] = self.scale_bar_checkbox.isChecked()
        self.config['display']['scale_bar_size'] = self.scale_bar_size_spin.value()
        self.config['display']['scale_bar_units'] = self.scale_bar_units_edit.text()
        self.config['display']['pixels_per_unit'] = self.pixels_per_unit_spin.value()

        # Update navigation settings
        self.config['navigation']['frame_rate'] = self.frame_rate_spin.value()
        self.config['navigation']['loop_playback'] = self.loop_checkbox.isChecked()
        self.config['navigation']['bidirectional_playback'] = self.bidirectional_checkbox.isChecked()
        self.config['navigation']['mouse_wheel_sensitivity'] = self.wheel_sensitivity_spin.value()

        # Update processing settings
        self.config['processing']['default_roi_color'] = color_map[self.roi_color_combo.currentIndex()]
        self.config['processing']['default_roi_width'] = self.roi_width_spin.value()
        self.config['processing']['smoothing_kernel_size'] = self.kernel_size_spin.value()
        self.config['processing']['registration_max_iterations'] = self.max_iterations_spin.value()
        self.config['processing']['registration_precision'] = self.precision_spin.value()

        # Update export settings
        self.config['export']['default_directory'] = self.export_dir_edit.text()
        self.config['export']['default_format'] = self.format_combo.currentText()
        self.config['export']['default_dpi'] = self.dpi_spin.value()
        self.config['export']['default_quality'] = self.quality_spin.value()
        self.config['export']['include_timestamp'] = self.timestamp_checkbox.isChecked()
        self.config['export']['include_analysis_info'] = self.analysis_info_checkbox.isChecked()

        # Update keyboard shortcuts
        for action in self.config['keyboard_shortcuts'].keys():
            shortcut_edit = self.findChild(QLineEdit, f"shortcut_{action}")
            if shortcut_edit:
                self.config['keyboard_shortcuts'][action] = shortcut_edit.text()

        # Emit settings changed signal if needed
        # This would be connected to a slot in the main window

    def reset_settings(self):
        """Reset settings to defaults."""
        # This would typically access the default settings from the config module
        # For now, we'll just close and reopen the dialog
        self.reject()

        # In a real implementation, you would:
        # 1. Reset to defaults
        # 2. Refresh the UI to reflect the defaults

    def get_settings(self):
        """Return the updated settings."""
        return self.config


class ContrastDialog(QDialog):
    """Dialog for adjusting contrast settings."""

    def __init__(self, image_stack, parent=None):
        """Initialize contrast dialog."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')
        self.image_stack = image_stack

        # Initialize with defaults
        self.p_low = 2
        self.p_high = 98

        # Setup UI
        self.setWindowTitle("Adjust Contrast")
        self.resize(400, 350)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create histogram
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create settings controls
        controls_layout = QGridLayout()

        controls_layout.addWidget(QLabel("Lower Percentile:"), 0, 0)
        self.low_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_slider.setMinimum(0)
        self.low_slider.setMaximum(50)
        self.low_slider.setValue(self.p_low)
        self.low_slider.valueChanged.connect(self.update_low_percentile)
        controls_layout.addWidget(self.low_slider, 0, 1)
        self.low_value_label = QLabel(f"{self.p_low}%")
        controls_layout.addWidget(self.low_value_label, 0, 2)

        controls_layout.addWidget(QLabel("Upper Percentile:"), 1, 0)
        self.high_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_slider.setMinimum(50)
        self.high_slider.setMaximum(100)
        self.high_slider.setValue(self.p_high)
        self.high_slider.valueChanged.connect(self.update_high_percentile)
        controls_layout.addWidget(self.high_slider, 1, 1)
        self.high_value_label = QLabel(f"{self.p_high}%")
        controls_layout.addWidget(self.high_value_label, 1, 2)

        # Add reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_values)
        controls_layout.addWidget(reset_button, 2, 0, 1, 3)

        layout.addLayout(controls_layout)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Plot initial histogram
        self.plot_histogram()

    def plot_histogram(self):
        """Plot image histogram with percentile lines."""
        # Clear the figure
        self.figure.clear()

        # Create axes
        ax = self.figure.add_subplot(111)

        # Get first frame as sample
        frame = self.image_stack.get_frame(0)

        # Compute histogram
        hist, bins = np.histogram(frame.flatten(), bins=256)

        # Plot histogram
        ax.bar(bins[:-1], hist, width=bins[1] - bins[0], color='blue', alpha=0.7)

        # Calculate percentile values
        p_low_val = np.percentile(frame, self.p_low)
        p_high_val = np.percentile(frame, self.p_high)

        # Plot percentile lines
        ax.axvline(p_low_val, color='red', linestyle='--', label=f'{self.p_low}%')
        ax.axvline(p_high_val, color='green', linestyle='--', label=f'{self.p_high}%')

        # Add labels and legend
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Image Histogram')
        ax.legend()

        # Refresh canvas
        self.canvas.draw()

    def update_low_percentile(self, value):
        """Update low percentile value."""
        self.p_low = value
        self.low_value_label.setText(f"{value}%")

        # Ensure high percentile is always greater
        if self.p_low >= self.high_slider.value():
            self.high_slider.setValue(self.p_low + 1)

        self.plot_histogram()

    def update_high_percentile(self, value):
        """Update high percentile value."""
        self.p_high = value
        self.high_value_label.setText(f"{value}%")

        # Ensure low percentile is always less
        if self.p_high <= self.low_slider.value():
            self.low_slider.setValue(self.p_high - 1)

        self.plot_histogram()

    def reset_values(self):
        """Reset to default values."""
        self.low_slider.setValue(2)
        self.high_slider.setValue(98)

    def get_contrast_params(self):
        """Return the contrast parameters."""
        return {
            'p_low': self.p_low,
            'p_high': self.p_high
        }


class ProjectionDialog(QDialog):
    """Dialog for creating Z-projections."""

    def __init__(self, parent=None):
        """Initialize projection dialog."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')

        # Setup UI
        self.setWindowTitle("Create Z-Projection")
        self.resize(350, 250)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Projection method
        method_group = QGroupBox("Projection Method")
        method_layout = QVBoxLayout()

        self.max_radio = QRadioButton("Maximum Intensity")
        self.max_radio.setChecked(True)
        method_layout.addWidget(self.max_radio)

        self.min_radio = QRadioButton("Minimum Intensity")
        method_layout.addWidget(self.min_radio)

        self.mean_radio = QRadioButton("Mean Intensity")
        method_layout.addWidget(self.mean_radio)

        self.median_radio = QRadioButton("Median Intensity")
        method_layout.addWidget(self.median_radio)

        self.std_radio = QRadioButton("Standard Deviation")
        method_layout.addWidget(self.std_radio)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Frame range
        range_group = QGroupBox("Frame Range")
        range_layout = QVBoxLayout()

        self.all_frames_radio = QRadioButton("All Frames")
        self.all_frames_radio.setChecked(True)
        range_layout.addWidget(self.all_frames_radio)

        range_layout.addWidget(QLabel("Or specify range:"))

        range_input_layout = QHBoxLayout()
        range_input_layout.addWidget(QLabel("From:"))
        self.from_spin = QSpinBox()
        self.from_spin.setMinimum(1)
        self.from_spin.setMaximum(9999)
        self.from_spin.setValue(1)
        range_input_layout.addWidget(self.from_spin)

        range_input_layout.addWidget(QLabel("To:"))
        self.to_spin = QSpinBox()
        self.to_spin.setMinimum(1)
        self.to_spin.setMaximum(9999)
        self.to_spin.setValue(10)
        range_input_layout.addWidget(self.to_spin)

        range_layout.addLayout(range_input_layout)

        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Save options
        self.save_checkbox = QCheckBox("Save Projection to File")
        layout.addWidget(self.save_checkbox)

        self.save_path_edit = QLineEdit()
        self.save_path_edit.setEnabled(False)
        layout.addWidget(self.save_path_edit)

        save_path_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self.browse_save_path)
        save_path_layout.addWidget(self.browse_button)
        layout.addLayout(save_path_layout)

        # Connect save checkbox
        self.save_checkbox.stateChanged.connect(self.toggle_save_options)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def toggle_save_options(self, state):
        """Toggle save options based on checkbox state."""
        enabled = state == Qt.CheckState.Checked
        self.save_path_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)

    def browse_save_path(self):
        """Browse for save file path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Projection", "", "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            self.save_path_edit.setText(file_path)

    def get_projection_params(self):
        """Return the projection parameters."""
        # Determine projection method
        if self.max_radio.isChecked():
            method = 'max'
        elif self.min_radio.isChecked():
            method = 'min'
        elif self.mean_radio.isChecked():
            method = 'mean'
        elif self.median_radio.isChecked():
            method = 'median'
        elif self.std_radio.isChecked():
            method = 'std'
        else:
            method = 'max'

        # Determine frame range
        if self.all_frames_radio.isChecked():
            frame_range = None
        else:
            frame_range = (self.from_spin.value() - 1, self.to_spin.value() - 1)

        # Determine save options
        save = self.save_checkbox.isChecked()
        save_path = self.save_path_edit.text() if save else None

        return {
            'method': method,
            'frame_range': frame_range,
            'save': save,
            'save_path': save_path
        }


class RegistrationDialog(QDialog):
    """Dialog for image registration."""

    def __init__(self, parent=None):
        """Initialize registration dialog."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')

        # Setup UI
        self.setWindowTitle("Register Images")
        self.resize(350, 300)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Registration stack selection
        stack_group = QGroupBox("Registration Target")
        stack_layout = QVBoxLayout()

        self.fluor_to_mask_radio = QRadioButton("Register Fluorescence to Mask")
        stack_layout.addWidget(self.fluor_to_mask_radio)

        self.mask_to_fluor_radio = QRadioButton("Register Mask to Fluorescence")
        self.mask_to_fluor_radio.setChecked(True)
        stack_layout.addWidget(self.mask_to_fluor_radio)

        self.self_registration_radio = QRadioButton("Register Fluorescence Stack to Reference Frame")
        stack_layout.addWidget(self.self_registration_radio)

        stack_group.setLayout(stack_layout)
        layout.addWidget(stack_group)

        # Registration method
        method_group = QGroupBox("Registration Method")
        method_layout = QVBoxLayout()

        self.phase_radio = QRadioButton("Phase Correlation (Translation Only)")
        self.phase_radio.setChecked(True)
        method_layout.addWidget(self.phase_radio)

        self.feature_radio = QRadioButton("Feature-based (Homography)")
        method_layout.addWidget(self.feature_radio)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Reference frame
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference Frame:"))
        self.ref_frame_spin = QSpinBox()
        self.ref_frame_spin.setMinimum(0)
        self.ref_frame_spin.setMaximum(9999)
        self.ref_frame_spin.setValue(0)
        ref_layout.addWidget(self.ref_frame_spin)
        layout.addLayout(ref_layout)

        # Precision
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Registration Precision:"))
        self.precision_spin = QDoubleSpinBox()
        self.precision_spin.setRange(0.01, 10.0)
        self.precision_spin.setSingleStep(0.1)
        self.precision_spin.setValue(1.0)
        precision_layout.addWidget(self.precision_spin)
        layout.addLayout(precision_layout)

        # Options
        self.preview_checkbox = QCheckBox("Preview Result")
        layout.addWidget(self.preview_checkbox)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_registration_params(self):
        """Return the registration parameters."""
        # Determine stack to register
        if self.fluor_to_mask_radio.isChecked():
            stack = 'fluorescence'
        elif self.mask_to_fluor_radio.isChecked():
            stack = 'mask'
        elif self.self_registration_radio.isChecked():
            stack = 'self'
        else:
            stack = 'mask'

        # Determine registration method
        if self.phase_radio.isChecked():
            method = 'phase'
        elif self.feature_radio.isChecked():
            method = 'feature'
        else:
            method = 'phase'

        return {
            'stack': stack,
            'method': method,
            'reference_frame': self.ref_frame_spin.value(),
            'precision': self.precision_spin.value(),
            'preview': self.preview_checkbox.isChecked()
        }


class AnalysisDialog(QDialog):
    """Dialog for configuring analysis."""

    def __init__(self, parent=None, analysis_type='intensity'):
        """Initialize analysis dialog."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')
        self.analysis_type = analysis_type

        # Setup UI
        self.setWindowTitle(f"{analysis_type.capitalize()} Analysis")
        self.resize(350, 300)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # ROI selection
        roi_group = QGroupBox("ROI Selection")
        roi_layout = QVBoxLayout()

        self.current_roi_radio = QRadioButton("Use Current ROIs")
        self.current_roi_radio.setChecked(True)
        roi_layout.addWidget(self.current_roi_radio)

        self.mask_roi_radio = QRadioButton("Use Mask as ROI")
        roi_layout.addWidget(self.mask_roi_radio)

        self.whole_image_radio = QRadioButton("Analyze Whole Image")
        roi_layout.addWidget(self.whole_image_radio)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # Add specific controls based on analysis type
        if self.analysis_type == 'intensity':
            # Intensity analysis options
            options_group = QGroupBox("Analysis Options")
            options_layout = QVBoxLayout()

            self.stats_checkbox = QCheckBox("Include Basic Statistics")
            self.stats_checkbox.setChecked(True)
            options_layout.addWidget(self.stats_checkbox)

            self.histogram_checkbox = QCheckBox("Include Histogram")
            self.histogram_checkbox.setChecked(True)
            options_layout.addWidget(self.histogram_checkbox)

            options_group.setLayout(options_layout)
            layout.addWidget(options_group)

        elif self.analysis_type == 'time_series':
            # Time series analysis options
            options_group = QGroupBox("Analysis Options")
            options_layout = QVBoxLayout()

            self.mean_checkbox = QCheckBox("Analyze Mean Intensity")
            self.mean_checkbox.setChecked(True)
            options_layout.addWidget(self.mean_checkbox)

            self.trend_checkbox = QCheckBox("Detect Trends")
            self.trend_checkbox.setChecked(True)
            options_layout.addWidget(self.trend_checkbox)

            self.peaks_checkbox = QCheckBox("Detect Peaks")
            self.peaks_checkbox.setChecked(True)
            options_layout.addWidget(self.peaks_checkbox)

            options_group.setLayout(options_layout)
            layout.addWidget(options_group)

        elif self.analysis_type == 'feature':
            # Feature detection options
            options_group = QGroupBox("Detection Options")
            options_layout = QGridLayout()

            # Detection method
            options_layout.addWidget(QLabel("Method:"), 0, 0)
            self.method_combo = QComboBox()
            self.method_combo.addItems(["Blob Detection", "Corner Detection"])
            options_layout.addWidget(self.method_combo, 0, 1)

            # Threshold
            options_layout.addWidget(QLabel("Threshold:"), 1, 0)
            self.threshold_spin = QDoubleSpinBox()
            self.threshold_spin.setRange(0.01, 1.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setValue(0.1)
            options_layout.addWidget(self.threshold_spin, 1, 1)

            # Size range
            options_layout.addWidget(QLabel("Min Size:"), 2, 0)
            self.min_size_spin = QDoubleSpinBox()
            self.min_size_spin.setRange(1.0, 20.0)
            self.min_size_spin.setSingleStep(0.5)
            self.min_size_spin.setValue(1.0)
            options_layout.addWidget(self.min_size_spin, 2, 1)

            options_layout.addWidget(QLabel("Max Size:"), 3, 0)
            self.max_size_spin = QDoubleSpinBox()
            self.max_size_spin.setRange(5.0, 50.0)
            self.max_size_spin.setSingleStep(0.5)
            self.max_size_spin.setValue(10.0)
            options_layout.addWidget(self.max_size_spin, 3, 1)

            options_group.setLayout(options_layout)
            layout.addWidget(options_group)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                    QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_analysis_params(self):
        """Return the analysis parameters."""
        # Get ROI selection type
        if self.current_roi_radio.isChecked():
            roi_type = 'current'
        elif self.mask_roi_radio.isChecked():
            roi_type = 'mask'
        else:
            roi_type = 'whole'

        # Create base parameters
        params = {
            'roi_type': roi_type
        }

        # Add analysis-specific parameters
        if self.analysis_type == 'intensity':
            params.update({
                'include_stats': self.stats_checkbox.isChecked(),
                'include_histogram': self.histogram_checkbox.isChecked()
            })

        elif self.analysis_type == 'time_series':
            params.update({
                'analyze_mean': self.mean_checkbox.isChecked(),
                'detect_trends': self.trend_checkbox.isChecked(),
                'detect_peaks': self.peaks_checkbox.isChecked()
            })

        elif self.analysis_type == 'feature':
            params.update({
                'method': self.method_combo.currentText().lower().split()[0],  # 'blob' or 'corner'
                'threshold': self.threshold_spin.value(),
                'min_size': self.min_size_spin.value(),
                'max_size': self.max_size_spin.value()
            })

        return params

class ExportDialog(QDialog):
    """Dialog for export options."""

    def __init__(self, parent=None):
        """Initialize export dialog."""
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.resize(400, 400)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Export type selection
        type_group = QGroupBox("Export Type")
        type_layout = QVBoxLayout()

        self.frames_radio = QRadioButton("Export Frames")
        self.frames_radio.setChecked(True)
        self.frames_radio.toggled.connect(self.update_ui_state)
        type_layout.addWidget(self.frames_radio)

        self.movie_radio = QRadioButton("Export Movie")
        self.movie_radio.toggled.connect(self.update_ui_state)
        type_layout.addWidget(self.movie_radio)

        self.data_radio = QRadioButton("Export Data")
        self.data_radio.toggled.connect(self.update_ui_state)
        type_layout.addWidget(self.data_radio)

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Frame export options
        self.frame_options = QGroupBox("Frame Export Options")
        frame_layout = QGridLayout()

        frame_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "tiff"])
        frame_layout.addWidget(self.format_combo, 0, 1)

        frame_layout.addWidget(QLabel("Frames:"), 1, 0)
        self.frames_combo = QComboBox()
        self.frames_combo.addItems(["All Frames", "Current Frame", "Frame Range"])
        self.frames_combo.currentTextChanged.connect(self.update_frame_range_state)
        frame_layout.addWidget(self.frames_combo, 1, 1)

        # Frame range inputs
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("From:"))
        self.frame_start = QSpinBox()
        self.frame_start.setMinimum(1)
        range_layout.addWidget(self.frame_start)
        range_layout.addWidget(QLabel("To:"))
        self.frame_end = QSpinBox()
        self.frame_end.setMinimum(1)
        range_layout.addWidget(self.frame_end)
        frame_layout.addLayout(range_layout, 2, 0, 1, 2)

        self.frame_options.setLayout(frame_layout)
        layout.addWidget(self.frame_options)

        # Movie export options
        self.movie_options = QGroupBox("Movie Export Options")
        movie_layout = QGridLayout()

        movie_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        movie_layout.addWidget(self.fps_spin, 0, 1)

        movie_layout.addWidget(QLabel("Quality:"), 1, 0)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        movie_layout.addWidget(self.quality_spin, 1, 1)

        self.movie_options.setLayout(movie_layout)
        layout.addWidget(self.movie_options)

        # Data export options
        self.data_options = QGroupBox("Data Export Options")
        data_layout = QVBoxLayout()

        self.include_stats = QCheckBox("Include Basic Statistics")
        self.include_stats.setChecked(True)
        data_layout.addWidget(self.include_stats)

        self.include_roi = QCheckBox("Include ROI Data")
        self.include_roi.setChecked(True)
        data_layout.addWidget(self.include_roi)

        self.data_options.setLayout(data_layout)
        layout.addWidget(self.data_options)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout()

        output_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.output_dir = QLineEdit()
        output_layout.addWidget(self.output_dir, 0, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output)
        output_layout.addWidget(browse_button, 0, 2)

        self.apply_overlay = QCheckBox("Apply Overlay")
        self.apply_overlay.setChecked(True)
        output_layout.addWidget(self.apply_overlay, 1, 0, 1, 3)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                    QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Initialize UI state
        self.update_ui_state()
        self.update_frame_range_state()

    def update_ui_state(self):
        """Update UI element states based on export type."""
        frames_selected = self.frames_radio.isChecked()
        movie_selected = self.movie_radio.isChecked()
        data_selected = self.data_radio.isChecked()

        self.frame_options.setVisible(frames_selected)
        self.movie_options.setVisible(movie_selected)
        self.data_options.setVisible(data_selected)
        self.apply_overlay.setVisible(frames_selected or movie_selected)

    def update_frame_range_state(self):
        """Update frame range input state."""
        range_enabled = self.frames_combo.currentText() == "Frame Range"
        self.frame_start.setEnabled(range_enabled)
        self.frame_end.setEnabled(range_enabled)

    def browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir.text()
        )
        if directory:
            self.output_dir.setText(directory)

    def get_export_options(self):
        """Return the export options."""
        # Determine export type
        if self.frames_radio.isChecked():
            export_type = 'frames'
        elif self.movie_radio.isChecked():
            export_type = 'movie'
        else:
            export_type = 'data'

        # Get frame selection
        frame_selection = self.frames_combo.currentText().lower()
        if frame_selection == "frame range":
            frames = f"{self.frame_start.value()}-{self.frame_end.value()}"
        elif frame_selection == "current frame":
            frames = "current"
        else:
            frames = "all"

        # Create options dictionary
        options = {
            'type': export_type,
            'output_dir': self.output_dir.text(),
            'apply_overlay': self.apply_overlay.isChecked()
        }

        # Add type-specific options
        if export_type == 'frames':
            options.update({
                'format': self.format_combo.currentText(),
                'frames': frames
            })
        elif export_type == 'movie':
            options.update({
                'fps': self.fps_spin.value(),
                'quality': self.quality_spin.value()
            })
        else:  # data
            options.update({
                'include_stats': self.include_stats.isChecked(),
                'include_roi': self.include_roi.isChecked()
            })

        return options


class ResultsDialog(QDialog):
    """Dialog for showing analysis results."""

    def __init__(self, title, subtitle, results, parent=None):
        """Initialize results dialog."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)

        self.results = results
        self.init_ui(subtitle)

    def init_ui(self, subtitle):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Add title and subtitle
        title_label = QLabel(self.windowTitle())
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 10pt; color: #666;")
        layout.addWidget(subtitle_label)

        # Create table for results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)

        # Populate table with results
        self.populate_results_table()

        layout.addWidget(self.results_table)

        # Add plot if applicable
        if isinstance(self.results, list):
            # For multiple ROIs, create plots
            self.figure = Figure(figsize=(6, 4))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            self.plot_results()

        # Add export button
        export_button = QPushButton("Export Results...")
        export_button.clicked.connect(self.export_results)
        layout.addWidget(export_button)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def populate_results_table(self):
        """Populate the results table."""
        if isinstance(self.results, dict):
            # Single set of results
            self.results_table.setRowCount(len(self.results))
            for i, (key, value) in enumerate(self.results.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(str(key)))
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                self.results_table.setItem(i, 1, QTableWidgetItem(value_str))
        elif isinstance(self.results, list):
            # Multiple ROIs
            total_rows = sum(len(roi_results) for roi_results in self.results if roi_results)
            self.results_table.setRowCount(total_rows)

            current_row = 0
            for i, roi_results in enumerate(self.results):
                if roi_results:
                    # Add ROI header
                    self.results_table.setItem(current_row, 0, QTableWidgetItem(f"ROI {i+1}"))
                    self.results_table.setItem(current_row, 1, QTableWidgetItem(""))
                    current_row += 1

                    # Add ROI results
                    for key, value in roi_results.items():
                        self.results_table.setItem(current_row, 0, QTableWidgetItem(str(key)))
                        if isinstance(value, float):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        self.results_table.setItem(current_row, 1, QTableWidgetItem(value_str))
                        current_row += 1

        self.results_table.resizeColumnsToContents()

    def plot_results(self):
        """Create plots for results if applicable."""
        if not isinstance(self.results, list):
            return

        # Clear figure
        self.figure.clear()

        # Create subplot
        ax = self.figure.add_subplot(111)

        # Plot data for each ROI
        for i, roi_results in enumerate(self.results):
            if roi_results and 'mean' in roi_results:
                label = f"ROI {i+1}"
                ax.plot(roi_results['mean'], label=label)

        # Add labels and legend
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mean Intensity')
        ax.set_title('ROI Intensity Profiles')
        ax.legend()
        ax.grid(True)

        # Refresh canvas
        self.canvas.draw()

    def export_results(self):
        """Export results to CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Write header
                    f.write("Metric,Value\n")

                    # Write data
                    if isinstance(self.results, dict):
                        for key, value in self.results.items():
                            f.write(f"{key},{value}\n")
                    elif isinstance(self.results, list):
                        for i, roi_results in enumerate(self.results):
                            if roi_results:
                                f.write(f"\nROI {i+1}\n")
                                for key, value in roi_results.items():
                                    f.write(f"{key},{value}\n")

                QMessageBox.information(self, "Success", "Results exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")


class TimeSeriesDialog(QDialog):
    """Dialog for showing time series analysis."""

    def __init__(self, results, parent=None):
        """Initialize time series dialog."""
        super().__init__(parent)
        self.setWindowTitle("Time Series Analysis")
        self.resize(800, 600)

        self.results = results
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        tab_widget = QTabWidget()

        # Add plots tab
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)

        # Create figure for plots
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)

        # Create plots
        self.create_plots()

        tab_widget.addTab(plots_tab, "Plots")

        # Add metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)

        # Create table for metrics
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["ROI", "Mean", "Std Dev", "Trend"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)

        # Populate metrics table
        self.populate_metrics_table()

        metrics_layout.addWidget(self.metrics_table)

        tab_widget.addTab(metrics_tab, "Metrics")

        # Add correlation tab if multiple ROIs
        if 'region_ids' in self.results and len(self.results['region_ids']) > 1:
            correlation_tab = QWidget()
            correlation_layout = QVBoxLayout(correlation_tab)

            # Create correlation matrix plot
            self.correlation_figure = Figure(figsize=(6, 6))
            self.correlation_canvas = FigureCanvas(self.correlation_figure)
            correlation_layout.addWidget(self.correlation_canvas)

            # Create correlation matrix
            self.create_correlation_matrix()

            tab_widget.addTab(correlation_tab, "Correlations")

        layout.addWidget(tab_widget)

        # Add export button
        export_button = QPushButton("Export Analysis...")
        export_button.clicked.connect(self.export_analysis)
        layout.addWidget(export_button)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def create_plots(self):
        """Create time series plots."""
        self.figure.clear()

        # Create intensity plot
        ax1 = self.figure.add_subplot(211)

        # Plot time series for each ROI
        if 'region_ids' in self.results:
            for i, roi_id in enumerate(self.results['region_ids']):
                if 'mean' in self.results and i < len(self.results['mean']):
                    ax1.plot(self.results['time'], self.results['mean'][i], label=roi_id)
        else:
            # Single ROI
            if 'mean' in self.results:
                ax1.plot(self.results['time'], self.results['mean'], label='ROI')

        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Mean Intensity')
        ax1.set_title('Intensity over Time')
        ax1.legend()
        ax1.grid(True)

        # Create derivative plot
        ax2 = self.figure.add_subplot(212)

        # Plot derivatives
        if 'region_ids' in self.results:
            for i, roi_id in enumerate(self.results['region_ids']):
                if 'mean' in self.results and i < len(self.results['mean']):
                    # Calculate derivative
                    derivative = np.diff(self.results['mean'][i])
                    ax2.plot(self.results['time'][1:], derivative, label=roi_id)
        else:
            # Single ROI
            if 'mean' in self.results:
                derivative = np.diff(self.results['mean'])
                ax2.plot(self.results['time'][1:], derivative, label='ROI')

        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Rate of Change')
        ax2.set_title('Intensity Change Rate')
        ax2.legend()
        ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def populate_metrics_table(self):
        """Populate the metrics table."""
        if 'region_ids' in self.results:
            # Multiple ROIs
            self.metrics_table.setRowCount(len(self.results['region_ids']))

            for i, roi_id in enumerate(self.results['region_ids']):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(roi_id))

                if 'mean' in self.results and i < len(self.results['mean']):
                    mean_value = np.mean(self.results['mean'][i])
                    std_value = np.std(self.results['mean'][i])

                    self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{mean_value:.4f}"))
                    self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{std_value:.4f}"))

                    # Calculate trend
                    slope, _, r_value, p_value, _ = stats.linregress(
                        range(len(self.results['mean'][i])),
                        self.results['mean'][i]
                    )

                    if p_value < 0.05:
                        trend = "Increasing" if slope > 0 else "Decreasing"
                        trend += f" (p={p_value:.4f}, R²={r_value**2:.4f})"
                    else:
                        trend = "No significant trend"

                    self.metrics_table.setItem(i, 3, QTableWidgetItem(trend))
        else:
            # Single ROI
            self.metrics_table.setRowCount(1)
            self.metrics_table.setItem(0, 0, QTableWidgetItem("ROI"))

            if 'mean' in self.results:
                mean_value = np.mean(self.results['mean'])
                std_value = np.std(self.results['mean'])

                self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{mean_value:.4f}"))
                self.metrics_table.setItem(0, 2, QTableWidgetItem(f"{std_value:.4f}"))

                # Calculate trend
                slope, _, r_value, p_value, _ = stats.linregress(
                    range(len(self.results['mean'])),
                    self.results['mean']
                )

                if p_value < 0.05:
                    trend = "Increasing" if slope > 0 else "Decreasing"
                    trend += f" (p={p_value:.4f}, R²={r_value**2:.4f})"
                else:
                    trend = "No significant trend"

                self.metrics_table.setItem(0, 3, QTableWidgetItem(trend))

        self.metrics_table.resizeColumnsToContents()

    def create_correlation_matrix(self):
        """Create correlation matrix plot."""
        if not ('region_ids' in self.results and 'mean' in self.results):
            return

        self.correlation_figure.clear()
        ax = self.correlation_figure.add_subplot(111)

        # Calculate correlation matrix
        data = np.array(self.results['mean'])
        corr_matrix = np.corrcoef(data)

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        self.correlation_figure.colorbar(im, ax=ax, label='Correlation')

        # Add labels
        roi_ids = self.results['region_ids']
        ax.set_xticks(range(len(roi_ids)))
        ax.set_yticks(range(len(roi_ids)))
        ax.set_xticklabels(roi_ids, rotation=45, ha='right')
        ax.set_yticklabels(roi_ids)

        # Add title
        ax.set_title('ROI Correlation Matrix')

        # Add correlation values
        for i in range(len(roi_ids)):
            for j in range(len(roi_ids)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black')

        self.correlation_figure.tight_layout()
        self.correlation_canvas.draw()

    def export_analysis(self):
        """Export analysis results."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Results", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Create DataFrame from results
            if 'region_ids' in self.results:
                # Multiple ROIs
                data = {}
                data['Frame'] = self.results['time']

                for i, roi_id in enumerate(self.results['region_ids']):
                    if 'mean' in self.results and i < len(self.results['mean']):
                        data[f'{roi_id}_Mean'] = self.results['mean'][i]
                    if 'std' in self.results and i < len(self.results['std']):
                        data[f'{roi_id}_StdDev'] = self.results['std'][i]
            else:
                # Single ROI
                data = {
                    'Frame': self.results['time'],
                    'Mean': self.results['mean']
                }
                if 'std' in self.results:
                    data['StdDev'] = self.results['std']

            df = pd.DataFrame(data)

            # Save based on file extension
            if file_path.lower().endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False)

            QMessageBox.information(self, "Success", "Analysis results exported successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export analysis results: {str(e)}")


class LogViewerDialog(QDialog):
    """Dialog for viewing log messages."""

    def __init__(self, parent=None):
        """Initialize log viewer dialog."""
        super().__init__(parent)
        self.setWindowTitle("Log Messages")
        self.resize(700, 500)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Log viewer placeholder"))

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)




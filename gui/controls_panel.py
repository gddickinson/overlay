"""
Controls panel for the Advanced TIFF Stack Viewer.
"""

import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QGridLayout,
                           QSlider, QLabel, QPushButton, QComboBox, QCheckBox,
                           QSpinBox, QDoubleSpinBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlsPanel(QWidget):
    """Panel containing controls for adjusting display and analysis settings."""

    # Custom signals
    display_settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the controls panel."""
        super().__init__(parent)

        self.logger = logging.getLogger('tiff_stack_viewer')

        # Initialize variables
        self.main_window = parent
        self.display_settings = {
            'fluorescence_visible': True,
            'mask_visible': True,
            'overlay_alpha': 0.5,
            'mask_color': [255, 0, 0],  # Default red
            'auto_contrast': True,
            'colormap': 'viridis'
        }

        # Set up UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create display tab
        self.display_tab = QWidget()
        self.setup_display_tab()
        self.tab_widget.addTab(self.display_tab, "Display")


        # Create processing tab
        self.processing_tab = QWidget()
        self.setup_processing_tab()
        self.tab_widget.addTab(self.processing_tab, "Processing")

        # Create analysis tab
        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "Analysis")

        # Add tab widget to layout
        layout.addWidget(self.tab_widget)

    def setup_display_tab(self):
        """Set up the display tab."""
        layout = QVBoxLayout(self.display_tab)

        # Visibility group
        visibility_group = QGroupBox("Visibility")
        visibility_layout = QGridLayout()

        # Fluorescence visibility
        self.fluor_checkbox = QCheckBox("Show Fluorescence")
        self.fluor_checkbox.setChecked(self.display_settings['fluorescence_visible'])
        self.fluor_checkbox.clicked.connect(self.handle_fluor_visibility)

        # Mask visibility
        self.mask_checkbox = QCheckBox("Show Mask")
        self.mask_checkbox.setChecked(self.display_settings['mask_visible'])
        self.mask_checkbox.clicked.connect(self.handle_mask_visibility)

        visibility_layout.addWidget(self.fluor_checkbox, 0, 0)
        visibility_layout.addWidget(self.mask_checkbox, 1, 0)
        visibility_group.setLayout(visibility_layout)

        # Overlay group
        overlay_group = QGroupBox("Overlay Settings")
        overlay_layout = QGridLayout()

        # Overlay alpha slider
        overlay_layout.addWidget(QLabel("Opacity:"), 0, 0)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.display_settings['overlay_alpha'] * 100))
        self.alpha_slider.valueChanged.connect(self.update_overlay_alpha)
        overlay_layout.addWidget(self.alpha_slider, 0, 1)
        self.alpha_value_label = QLabel(f"{self.display_settings['overlay_alpha']:.2f}")
        overlay_layout.addWidget(self.alpha_value_label, 0, 2)

        # Mask color selection
        overlay_layout.addWidget(QLabel("Mask Color:"), 1, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"])
        self.color_combo.currentIndexChanged.connect(self.update_mask_color)
        overlay_layout.addWidget(self.color_combo, 1, 1, 1, 2)

        overlay_group.setLayout(overlay_layout)

        # Color mapping group
        color_group = QGroupBox("Color Mapping")
        color_layout = QGridLayout()

        # Auto contrast
        self.auto_contrast_checkbox = QCheckBox("Auto Contrast")
        self.auto_contrast_checkbox.setChecked(self.display_settings['auto_contrast'])
        self.auto_contrast_checkbox.stateChanged.connect(self.update_auto_contrast)
        color_layout.addWidget(self.auto_contrast_checkbox, 0, 0, 1, 2)

        # Colormap selection
        color_layout.addWidget(QLabel("Colormap:"), 1, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis",
                                    "gray", "hot", "cool", "jet", "rainbow"])
        self.colormap_combo.setCurrentText(self.display_settings['colormap'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        color_layout.addWidget(self.colormap_combo, 1, 1)

        color_group.setLayout(color_layout)

        # Add groups to layout
        layout.addWidget(visibility_group)
        layout.addWidget(overlay_group)
        layout.addWidget(color_group)
        layout.addStretch()

    def setup_processing_tab(self):
        """Set up the processing tab."""
        layout = QVBoxLayout(self.processing_tab)

        # Filters group
        filters_group = QGroupBox("Apply Filter")
        filters_layout = QGridLayout()

        # Filter type selection
        filters_layout.addWidget(QLabel("Filter Type:"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Gaussian", "Median", "Bilateral", "Sobel", "Laplacian"])
        filters_layout.addWidget(self.filter_combo, 0, 1)

        # Filter parameters
        filters_layout.addWidget(QLabel("Parameter:"), 1, 0)
        self.filter_param_spin = QDoubleSpinBox()
        self.filter_param_spin.setRange(0.1, 10.0)
        self.filter_param_spin.setSingleStep(0.1)
        self.filter_param_spin.setValue(1.0)
        filters_layout.addWidget(self.filter_param_spin, 1, 1)

        # Apply button
        self.apply_filter_button = QPushButton("Apply")
        self.apply_filter_button.clicked.connect(self.apply_filter)
        filters_layout.addWidget(self.apply_filter_button, 2, 0, 1, 2)

        filters_group.setLayout(filters_layout)

        # Registration group
        registration_group = QGroupBox("Image Registration")
        registration_layout = QGridLayout()

        # Registration method
        registration_layout.addWidget(QLabel("Method:"), 0, 0)
        self.registration_combo = QComboBox()
        self.registration_combo.addItems(["Phase Correlation", "Feature-based"])
        registration_layout.addWidget(self.registration_combo, 0, 1)

        # Reference frame
        registration_layout.addWidget(QLabel("Reference Frame:"), 1, 0)
        self.ref_frame_spin = QSpinBox()
        self.ref_frame_spin.setMinimum(0)
        self.ref_frame_spin.setMaximum(999)
        self.ref_frame_spin.setValue(0)
        registration_layout.addWidget(self.ref_frame_spin, 1, 1)

        # Apply button
        self.apply_registration_button = QPushButton("Register Images")
        self.apply_registration_button.clicked.connect(self.apply_registration)
        registration_layout.addWidget(self.apply_registration_button, 2, 0, 1, 2)

        registration_group.setLayout(registration_layout)

        # Projection group
        projection_group = QGroupBox("Z-Projection")
        projection_layout = QGridLayout()

        # Projection method
        projection_layout.addWidget(QLabel("Method:"), 0, 0)
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["Maximum", "Minimum", "Mean", "Median", "Standard Deviation"])
        projection_layout.addWidget(self.projection_combo, 0, 1)

        # Frame range
        projection_layout.addWidget(QLabel("Frame Range:"), 1, 0)
        self.projection_range = QLabel("All Frames")
        projection_layout.addWidget(self.projection_range, 1, 1)

        # Create button
        self.create_projection_button = QPushButton("Create Projection")
        self.create_projection_button.clicked.connect(self.create_projection)
        projection_layout.addWidget(self.create_projection_button, 2, 0, 1, 2)

        projection_group.setLayout(projection_layout)

        # Add groups to layout
        layout.addWidget(filters_group)
        layout.addWidget(registration_group)
        layout.addWidget(projection_group)
        layout.addStretch()

    def setup_analysis_tab(self):
        """Set up the analysis tab."""
        layout = QVBoxLayout(self.analysis_tab)

        # ROI analysis group
        roi_group = QGroupBox("ROI Analysis")
        roi_layout = QGridLayout()

        # Analysis type
        roi_layout.addWidget(QLabel("Analysis Type:"), 0, 0)
        self.roi_analysis_combo = QComboBox()
        self.roi_analysis_combo.addItems(["Intensity", "Time Series"])
        roi_layout.addWidget(self.roi_analysis_combo, 0, 1)

        # ROI selection
        roi_layout.addWidget(QLabel("ROI Source:"), 1, 0)
        self.roi_source_combo = QComboBox()
        self.roi_source_combo.addItems(["Current ROIs", "Mask", "Whole Image"])
        roi_layout.addWidget(self.roi_source_combo, 1, 1)

        # Analyze button
        self.analyze_roi_button = QPushButton("Analyze")
        self.analyze_roi_button.clicked.connect(self.analyze_roi)
        roi_layout.addWidget(self.analyze_roi_button, 2, 0, 1, 2)

        roi_group.setLayout(roi_layout)

        # Feature detection group
        feature_group = QGroupBox("Feature Detection")
        feature_layout = QGridLayout()

        # Detection method
        feature_layout.addWidget(QLabel("Method:"), 0, 0)
        self.feature_method_combo = QComboBox()
        self.feature_method_combo.addItems(["Blob Detection", "Corner Detection"])
        feature_layout.addWidget(self.feature_method_combo, 0, 1)

        # Threshold
        feature_layout.addWidget(QLabel("Threshold:"), 1, 0)
        self.feature_threshold_spin = QDoubleSpinBox()
        self.feature_threshold_spin.setRange(0.01, 1.0)
        self.feature_threshold_spin.setSingleStep(0.01)
        self.feature_threshold_spin.setValue(0.1)
        feature_layout.addWidget(self.feature_threshold_spin, 1, 1)

        # Size range
        feature_layout.addWidget(QLabel("Min Size:"), 2, 0)
        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(1.0, 20.0)
        self.min_size_spin.setSingleStep(0.5)
        self.min_size_spin.setValue(1.0)
        feature_layout.addWidget(self.min_size_spin, 2, 1)

        feature_layout.addWidget(QLabel("Max Size:"), 3, 0)
        self.max_size_spin = QDoubleSpinBox()
        self.max_size_spin.setRange(5.0, 50.0)
        self.max_size_spin.setSingleStep(0.5)
        self.max_size_spin.setValue(10.0)
        feature_layout.addWidget(self.max_size_spin, 3, 1)

        # Detect button
        self.detect_features_button = QPushButton("Detect Features")
        self.detect_features_button.clicked.connect(self.detect_features)
        feature_layout.addWidget(self.detect_features_button, 4, 0, 1, 2)

        feature_group.setLayout(feature_layout)

        # Export group
        export_group = QGroupBox("Export")
        export_layout = QGridLayout()

        # Export type
        export_layout.addWidget(QLabel("Export Type:"), 0, 0)
        self.export_type_combo = QComboBox()
        self.export_type_combo.addItems(["Current Frame", "All Frames", "Movie", "Data"])
        export_layout.addWidget(self.export_type_combo, 0, 1)

        # Export button
        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export_data)
        export_layout.addWidget(self.export_button, 1, 0, 1, 2)

        export_group.setLayout(export_layout)

        # Add groups to layout
        layout.addWidget(roi_group)
        layout.addWidget(feature_group)
        layout.addWidget(export_group)
        layout.addStretch()

    # Signal handlers for display settings
    def update_fluorescence_visibility(self, state):
        """Update fluorescence visibility setting."""
        self.display_settings['fluorescence_visible'] = state == Qt.CheckState.Checked
        self.emit_display_settings()

    def update_mask_visibility(self, state):
        """Update mask visibility setting."""
        self.display_settings['mask_visible'] = state == Qt.CheckState.Checked
        self.emit_display_settings()

    def update_overlay_alpha(self, value):
        """Update overlay alpha value."""
        alpha = value / 100.0
        self.display_settings['overlay_alpha'] = alpha
        self.alpha_value_label.setText(f"{alpha:.2f}")
        self.emit_display_settings()

    def update_mask_color(self, index):
        """Update mask color setting."""
        colors = {
            0: [255, 0, 0],     # Red
            1: [0, 255, 0],     # Green
            2: [0, 0, 255],     # Blue
            3: [255, 255, 0],   # Yellow
            4: [255, 0, 255],   # Magenta
            5: [0, 255, 255]    # Cyan
        }
        self.display_settings['mask_color'] = colors[index]
        self.emit_display_settings()

    def update_auto_contrast(self, state):
        """Update auto contrast setting."""
        self.display_settings['auto_contrast'] = state == Qt.CheckState.Checked
        self.emit_display_settings()

    def update_colormap(self, colormap):
        """Update colormap setting."""
        self.display_settings['colormap'] = colormap
        self.emit_display_settings()

    # Processing functions
    def apply_filter(self):
        """Apply selected filter to current frame."""
        filter_type = self.filter_combo.currentText().lower()
        parameter = self.filter_param_spin.value()

        # Forward to main window
        if hasattr(self.main_window, 'apply_filter'):
            self.main_window.apply_filter(filter_type, parameter)

    def apply_registration(self):
        """Apply image registration."""
        method = self.registration_combo.currentText()
        reference_frame = self.ref_frame_spin.value()

        # Convert method name to code
        if method == "Phase Correlation":
            method_code = "phase"
        elif method == "Feature-based":
            method_code = "feature"
        else:
            method_code = "phase"

        # Forward to main window
        if hasattr(self.main_window, 'show_registration_dialog'):
            self.main_window.show_registration_dialog(method_code, reference_frame)

    def create_projection(self):
        """Create Z-projection."""
        method = self.projection_combo.currentText().lower()

        # Forward to main window
        if hasattr(self.main_window, 'show_projection_dialog'):
            self.main_window.show_projection_dialog(method)

    # Analysis functions
    def analyze_roi(self):
        """Analyze ROIs."""
        analysis_type = self.roi_analysis_combo.currentText().lower()
        roi_source = self.roi_source_combo.currentText().lower()

        # Convert roi source to code
        if roi_source == "Current ROIs":
            roi_type = "current"
        elif roi_source == "Mask":
            roi_type = "mask"
        else:
            roi_type = "whole"

        # Forward to main window
        if analysis_type == "intensity" and hasattr(self.main_window, 'show_intensity_analysis_dialog'):
            self.main_window.show_intensity_analysis_dialog(roi_type)
        elif analysis_type == "time series" and hasattr(self.main_window, 'show_time_series_dialog'):
            self.main_window.show_time_series_dialog(roi_type)

    def detect_features(self):
        """Detect features in current frame."""
        method = self.feature_method_combo.currentText().lower()
        threshold = self.feature_threshold_spin.value()
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()

        # Convert method name to code
        if method == "blob detection":
            method_code = "blob"
        elif method == "corner detection":
            method_code = "corner"
        else:
            method_code = "blob"

        # Forward to main window
        if hasattr(self.main_window, 'show_feature_detection_dialog'):
            self.main_window.show_feature_detection_dialog(method_code, threshold, min_size, max_size)

    def export_data(self):
        """Export data."""
        export_type = self.export_type_combo.currentText().lower()

        # Forward to main window
        if export_type == "current frame" and hasattr(self.main_window, 'save_current_frame'):
            self.main_window.save_current_frame()
        elif hasattr(self.main_window, 'show_export_dialog'):
            self.main_window.show_export_dialog(export_type)

    def emit_display_settings(self):
        """Emit the current display settings."""
        # Add min/max levels from histogram if available
        if hasattr(self.main_window, 'histogram'):
            min_level, max_level = self.main_window.histogram.get_levels()
            self.display_settings['min_level'] = min_level
            self.display_settings['max_level'] = max_level

        self.display_settings_changed.emit(self.display_settings)

    # def toggle_fluorescence(self, visible):
    #     """Toggle fluorescence visibility."""
    #     self.display_settings['fluorescence_visible'] = visible
    #     self.display_settings_changed.emit(self.display_settings)

    # def toggle_mask(self, visible):
    #     """Toggle mask visibility."""
    #     self.display_settings['mask_visible'] = visible
    #     self.display_settings_changed.emit(self.display_settings)

    # def on_fluorescence_visibility_changed(self, state):
    #     """Handle changes to fluorescence visibility."""
    #     self.display_settings['fluorescence_visible'] = (state == Qt.CheckState.Checked)
    #     self.display_settings_changed.emit(self.display_settings)

    # def on_mask_visibility_changed(self, state):
    #     """Handle changes to mask visibility."""
    #     self.display_settings['mask_visible'] = (state == Qt.CheckState.Checked)
    #     self.display_settings_changed.emit(self.display_settings)

    def handle_fluor_visibility(self, checked):
        """Handle fluorescence visibility change."""
        self.logger.debug(f"Fluorescence visibility changed to: {checked}")
        self.display_settings['fluorescence_visible'] = checked
        settings = dict(self.display_settings)  # Make a copy
        self.display_settings_changed.emit(settings)

    def handle_mask_visibility(self, checked):
        """Handle mask visibility change."""
        self.logger.debug(f"Mask visibility changed to: {checked}")
        self.display_settings['mask_visible'] = checked
        settings = dict(self.display_settings)  # Make a copy
        self.display_settings_changed.emit(settings)

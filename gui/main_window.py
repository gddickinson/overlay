"""
Main window for the Advanced TIFF Stack Viewer application.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import cv2

from PyQt6.QtWidgets import (QMainWindow, QApplication, QFileDialog, QMessageBox,
                            QSplitter, QStatusBar, QMenuBar, QMenu, QToolBar,
                            QDockWidget, QWidget, QVBoxLayout, QLabel)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence

from core.image_loader import ImageLoader, ImageStack
from core.image_processor import ImageProcessor
from core.data_analyzer import DataAnalyzer
from core.export_manager import ExportManager

from gui.image_view import EnhancedImageView
from gui.controls_panel import ControlsPanel
from gui.dialogs import (SettingsDialog, ContrastDialog, ProjectionDialog,
                        RegistrationDialog, AnalysisDialog, ExportDialog, ResultsDialog)
from gui.widgets.histogram_widget import HistogramWidget
from gui.widgets.roi_tools import ROIToolbar
from gui.widgets.navigation_bar import NavigationBar
from gui.widgets.timeline_widget import TimelineWidget


class MainWindow(QMainWindow):
    """Main application window."""

    # Custom signals
    frame_changed = pyqtSignal(int)
    fluorescence_loaded = pyqtSignal(object)
    mask_loaded = pyqtSignal(object)
    display_settings_changed = pyqtSignal(dict)
    roi_created = pyqtSignal(dict)
    roi_modified = pyqtSignal(dict)
    roi_deleted = pyqtSignal(str)
    analysis_completed = pyqtSignal(dict)

    def __init__(self, config, logger=None):
        """Initialize main window."""
        super().__init__()

        self.logger = logger or logging.getLogger('tiff_stack_viewer')
        self.config = config

        # Initialize core components
        self.image_loader = ImageLoader(self.logger)
        self.image_processor = ImageProcessor(self.logger)
        self.data_analyzer = DataAnalyzer(self.logger)
        self.export_manager = ExportManager(self.logger)

        # Initialize data containers
        self.fluorescence_stack = None
        self.mask_stack = None
        self.current_frame = 0
        self.display_settings = {
            'fluorescence_visible': True,
            'mask_visible': True,
            'overlay_alpha': 0.5,
            'mask_color': self.config['display']['default_mask_color'],
            'auto_contrast': self.config['display']['auto_contrast'],
            'colormap': self.config['display']['colormap'],
            'zoom_level': 1.0
        }
        self.rois = {}

        # Set up UI
        self.init_ui()

        # Connect signals
        self.connect_signals()

        # Restore window state
        self.restore_window_state()

        self.logger.info("Main window initialized")

    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Advanced TIFF Stack Viewer")
        self.resize(self.config['appearance']['window_size'][0],
                   self.config['appearance']['window_size'][1])
        self.move(self.config['appearance']['window_position'][0],
                 self.config['appearance']['window_position'][1])

        # Create central widget with image view
        self.image_view = EnhancedImageView(self)

        # Create navigation bar
        self.navigation_bar = NavigationBar(self)

        # Create timeline widget
        self.timeline = TimelineWidget(self)

        # Create main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.image_view)

        # Create bottom panel for timeline and navigation
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.addWidget(self.navigation_bar)
        bottom_layout.addWidget(self.timeline)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.main_splitter.addWidget(bottom_panel)
        self.main_splitter.setStretchFactor(0, 4)  # Give more space to image view

        # Set the main splitter as central widget
        self.setCentralWidget(self.main_splitter)

        # Create dock widgets
        self.create_dock_widgets()

        # Create menu and toolbar
        self.create_menu()
        self.create_toolbar()

        # Create status bar
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def create_dock_widgets(self):
        """Create dock widgets for controls and analysis."""
        # Controls panel
        self.controls_dock = QDockWidget("Controls", self)
        self.controls_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                          Qt.DockWidgetArea.RightDockWidgetArea)
        self.controls_panel = ControlsPanel(self)
        self.controls_dock.setWidget(self.controls_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)

        # Histogram panel
        self.histogram_dock = QDockWidget("Histogram", self)
        self.histogram_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                           Qt.DockWidgetArea.RightDockWidgetArea |
                                           Qt.DockWidgetArea.BottomDockWidgetArea)
        self.histogram = HistogramWidget(self)
        self.histogram_dock.setWidget(self.histogram)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.histogram_dock)

        # ROI tools
        self.roi_dock = QDockWidget("ROI Tools", self)
        self.roi_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                                     Qt.DockWidgetArea.RightDockWidgetArea |
                                     Qt.DockWidgetArea.TopDockWidgetArea)
        self.roi_toolbar = ROIToolbar(self)
        self.roi_dock.setWidget(self.roi_toolbar)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.roi_dock)

        # Tabify dock widgets on the right
        self.tabifyDockWidget(self.controls_dock, self.histogram_dock)
        self.controls_dock.raise_()

    def create_menu(self):
        """Create application menu."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Open actions
        open_fluor_action = QAction("Open &Fluorescence Stack", self)
        open_fluor_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['open_fluorescence']))
        open_fluor_action.triggered.connect(self.open_fluorescence_dialog)
        file_menu.addAction(open_fluor_action)

        open_mask_action = QAction("Open &Mask Stack", self)
        open_mask_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['open_mask']))
        open_mask_action.triggered.connect(self.open_mask_dialog)
        file_menu.addAction(open_mask_action)

        file_menu.addSeparator()

        # Recent files submenu
        self.recent_menu = QMenu("&Recent Files", self)
        file_menu.addMenu(self.recent_menu)
        self.update_recent_files_menu()

        file_menu.addSeparator()

        # Save/export actions
        save_action = QAction("&Save Current Frame", self)
        save_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['save']))
        save_action.triggered.connect(self.save_current_frame)
        file_menu.addAction(save_action)

        export_action = QAction("&Export...", self)
        export_action.triggered.connect(self.show_export_dialog)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        # ROI actions
        clear_rois_action = QAction("Clear All ROIs", self)
        clear_rois_action.triggered.connect(self.clear_rois)
        edit_menu.addAction(clear_rois_action)

        edit_menu.addSeparator()

        # Settings action
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        edit_menu.addAction(settings_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Zoom actions
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['zoom_in']))
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['zoom_out']))
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction("&Reset Zoom", self)
        reset_zoom_action.setShortcut(QKeySequence(self.config['keyboard_shortcuts']['reset_view']))
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)

        view_menu.addSeparator()

        # Contrast action
        contrast_action = QAction("Adjust &Contrast...", self)
        contrast_action.triggered.connect(self.show_contrast_dialog)
        view_menu.addAction(contrast_action)

        # Dock widget visibility
        view_menu.addSeparator()
        view_menu.addAction(self.controls_dock.toggleViewAction())
        view_menu.addAction(self.histogram_dock.toggleViewAction())
        view_menu.addAction(self.roi_dock.toggleViewAction())

        # View logs action
        view_menu.addSeparator()
        view_logs_action = QAction("Show &Log", self)
        view_logs_action.triggered.connect(self.show_log_dialog)
        view_menu.addAction(view_logs_action)

        # Image menu
        image_menu = menubar.addMenu("&Image")

        # Processing actions
        filter_menu = QMenu("Apply &Filter", self)

        gaussian_action = QAction("&Gaussian Blur", self)
        gaussian_action.triggered.connect(lambda: self.apply_filter("gaussian"))
        filter_menu.addAction(gaussian_action)

        median_action = QAction("&Median Filter", self)
        median_action.triggered.connect(lambda: self.apply_filter("median"))
        filter_menu.addAction(median_action)

        sobel_action = QAction("&Edge Detection", self)
        sobel_action.triggered.connect(lambda: self.apply_filter("sobel"))
        filter_menu.addAction(sobel_action)

        image_menu.addMenu(filter_menu)

        # Projection action
        projection_action = QAction("Create &Z-Projection...", self)
        projection_action.triggered.connect(self.show_projection_dialog)
        image_menu.addAction(projection_action)

        # Registration action
        registration_action = QAction("&Register Images...", self)
        registration_action.triggered.connect(self.show_registration_dialog)
        image_menu.addAction(registration_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        # Intensity analysis
        intensity_action = QAction("&Intensity Analysis...", self)
        intensity_action.triggered.connect(self.show_intensity_analysis_dialog)
        analysis_menu.addAction(intensity_action)

        # Time series analysis
        time_series_action = QAction("&Time Series Analysis...", self)
        time_series_action.triggered.connect(self.show_time_series_dialog)
        analysis_menu.addAction(time_series_action)

        # Feature detection
        feature_action = QAction("&Feature Detection...", self)
        feature_action.triggered.connect(self.show_feature_detection_dialog)
        analysis_menu.addAction(feature_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # Documentation action
        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

    def create_toolbar(self):
        """Create main toolbar."""
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(True)
        self.toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(self.toolbar)

        # Add navigation controls
        self.toolbar.addAction(QAction("First", self, triggered=self.navigation_bar.goto_first))
        self.toolbar.addAction(QAction("Previous", self, triggered=self.navigation_bar.goto_previous))
        self.toolbar.addAction(QAction("Play/Pause", self, triggered=self.navigation_bar.toggle_playback))
        self.toolbar.addAction(QAction("Next", self, triggered=self.navigation_bar.goto_next))
        self.toolbar.addAction(QAction("Last", self, triggered=self.navigation_bar.goto_last))

        self.toolbar.addSeparator()

        # Add view controls
        self.toolbar.addAction(QAction("Zoom In", self, triggered=self.zoom_in))
        self.toolbar.addAction(QAction("Zoom Out", self, triggered=self.zoom_out))
        self.toolbar.addAction(QAction("Reset View", self, triggered=self.reset_zoom))

    def toggle_fluorescence(self, checked):
        """Handle fluorescence toggle."""
        self.display_settings['fluorescence_visible'] = checked
        self.controls_panel.fluor_checkbox.setChecked(checked)  # Keep checkbox in sync

        # Update display with current frame data
        fluor_frame = None
        if checked and self.fluorescence_stack:
            fluor_frame = self.fluorescence_stack.get_frame(self.current_frame)

        mask_frame = None
        if self.display_settings['mask_visible'] and self.mask_stack:
            mask_frame = self.mask_stack.get_frame(self.current_frame)

        self.image_view.update_image(
            fluor_frame,
            mask_frame,
            self.display_settings['overlay_alpha'],
            self.display_settings['mask_color'],
            dict(self.display_settings)
        )

        # Update histogram
        if fluor_frame is not None and checked:
            self.histogram.update_histogram(fluor_frame)

    def toggle_mask(self, checked):
        """Handle mask toggle."""
        self.display_settings['mask_visible'] = checked
        self.controls_panel.mask_checkbox.setChecked(checked)  # Keep checkbox in sync

        # Update display with current frame data
        fluor_frame = None
        if self.display_settings['fluorescence_visible'] and self.fluorescence_stack:
            fluor_frame = self.fluorescence_stack.get_frame(self.current_frame)

        mask_frame = None
        if checked and self.mask_stack:
            mask_frame = self.mask_stack.get_frame(self.current_frame)

        self.image_view.update_image(
            fluor_frame,
            mask_frame,
            self.display_settings['overlay_alpha'],
            self.display_settings['mask_color'],
            dict(self.display_settings)
        )

    def connect_signals(self):
        """Connect signals and slots."""
        # Frame navigation signals
        self.frame_changed.connect(self.update_frame)
        self.navigation_bar.frame_changed.connect(self.frame_changed.emit)
        self.timeline.frame_changed.connect(self.frame_changed.emit)

        # Image data signals
        self.fluorescence_loaded.connect(self.on_fluorescence_loaded)
        self.mask_loaded.connect(self.on_mask_loaded)

        # Display settings signals
        self.controls_panel.display_settings_changed.connect(self.on_display_settings_changed)
        self.histogram.levels_changed.connect(self.update_levels)

        # ROI signals from ImageView
        self.image_view.roi_created.connect(self.handle_roi_created)
        self.image_view.roi_modified.connect(self.handle_roi_modified)
        self.image_view.roi_deleted.connect(self.handle_roi_deleted)

        # ROI signals from toolbar
        self.roi_toolbar.roi_delete_requested.connect(self.delete_roi)
        self.roi_toolbar.roi_clear_requested.connect(self.clear_rois)
        self.roi_toolbar.roi_analysis_requested.connect(self.analyze_roi)
        self.roi_toolbar.roi_selected.connect(self.select_roi)
        self.roi_toolbar.roi_type_changed.connect(self.image_view.set_roi_type)


    def restore_window_state(self):
        """Restore window state from settings."""
        settings = QSettings("MicroscopyLab", "TiffStackViewer")

        # Restore window geometry
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))

        # Restore window state (dock widgets, toolbars)
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))

        # Check if should maximize
        if self.config['appearance']['maximize_on_start']:
            self.showMaximized()

    def save_window_state(self):
        """Save window state to settings."""
        settings = QSettings("MicroscopyLab", "TiffStackViewer")

        # Save window geometry
        settings.setValue("geometry", self.saveGeometry())

        # Save window state (dock widgets, toolbars)
        settings.setValue("windowState", self.saveState())

        # Update config with current window position and size
        if not self.isMaximized() and not self.isFullScreen():
            self.config['appearance']['window_size'] = [self.width(), self.height()]
            self.config['appearance']['window_position'] = [self.x(), self.y()]

    def closeEvent(self, event):
        """Handle window close event."""
        # Save window state
        self.save_window_state()

        event.accept()

    # File operations
    def open_fluorescence_dialog(self):
        """Open dialog to select fluorescence stack."""
        file_filter = self.image_loader.get_supported_formats_filter()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Fluorescence Stack", "", file_filter
        )

        if file_path:
            self.load_fluorescence_stack(file_path)

    def open_mask_dialog(self):
        """Open dialog to select mask stack."""
        file_filter = self.image_loader.get_supported_formats_filter()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask Stack", "", file_filter
        )

        if file_path:
            self.load_mask_stack(file_path)

    def load_fluorescence_stack(self, file_path):
        """Load fluorescence stack from file."""
        self.logger.info(f"Loading fluorescence stack from {file_path}")
        self.statusBar.showMessage(f"Loading fluorescence stack from {file_path}...")

        # Load the stack
        self.fluorescence_stack = self.image_loader.load_file(file_path)

        if self.fluorescence_stack:
            # Update recent files list
            self.add_recent_file('fluorescence', file_path)

            # Reset current frame
            self.current_frame = 0

            # Emit signal
            self.fluorescence_loaded.emit(self.fluorescence_stack)

            self.statusBar.showMessage(f"Loaded fluorescence stack with {self.fluorescence_stack.max_frames} frames", 5000)
        else:
            self.statusBar.showMessage("Failed to load fluorescence stack", 5000)
            QMessageBox.critical(self, "Error", f"Failed to load fluorescence stack from {file_path}")

    def load_mask_stack(self, file_path):
        """Load mask stack from file."""
        self.logger.info(f"Loading mask stack from {file_path}")
        self.statusBar.showMessage(f"Loading mask stack from {file_path}...")

        # Load the stack
        self.mask_stack = self.image_loader.load_file(file_path)

        if self.mask_stack:
            # Update recent files list
            self.add_recent_file('mask', file_path)

            # Emit signal
            self.mask_loaded.emit(self.mask_stack)

            self.statusBar.showMessage(f"Loaded mask stack with {self.mask_stack.max_frames} frames", 5000)
        else:
            self.statusBar.showMessage("Failed to load mask stack", 5000)
            QMessageBox.critical(self, "Error", f"Failed to load mask stack from {file_path}")

    def add_recent_file(self, file_type, file_path):
        """Add file to recent files list."""
        recent_files = self.config['recent_files'][file_type]

        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)

        # Add to beginning of list
        recent_files.insert(0, file_path)

        # Limit to 10 entries
        self.config['recent_files'][file_type] = recent_files[:10]

        # Update menu
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        """Update recent files menu."""
        self.recent_menu.clear()

        # Add fluorescence recent files
        if self.config['recent_files']['fluorescence']:
            self.recent_menu.addSection("Fluorescence")
            for file_path in self.config['recent_files']['fluorescence']:
                action = QAction(os.path.basename(file_path), self)
                action.setStatusTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.load_fluorescence_stack(path))
                self.recent_menu.addAction(action)

        # Add mask recent files
        if self.config['recent_files']['mask']:
            self.recent_menu.addSection("Mask")
            for file_path in self.config['recent_files']['mask']:
                action = QAction(os.path.basename(file_path), self)
                action.setStatusTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.load_mask_stack(path))
                self.recent_menu.addAction(action)

        # Add clear action if there are recent files
        if self.config['recent_files']['fluorescence'] or self.config['recent_files']['mask']:
            self.recent_menu.addSeparator()
            clear_action = QAction("Clear Recent Files", self)
            clear_action.triggered.connect(self.clear_recent_files)
            self.recent_menu.addAction(clear_action)

    def clear_recent_files(self):
        """Clear recent files list."""
        self.config['recent_files']['fluorescence'] = []
        self.config['recent_files']['mask'] = []
        self.update_recent_files_menu()

    def save_current_frame(self):
        """Save current frame to file."""
        if not self.fluorescence_stack and not self.mask_stack:
            QMessageBox.warning(self, "Warning", "No data to save")
            return

        # Get current displayed image
        current_image = self.image_view.get_current_image()
        if current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return

        # Show save dialog
        file_filter = "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff);;All Files (*)"
        default_dir = self.config['export']['default_directory'] or os.path.expanduser("~")

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Current Frame", default_dir, file_filter
        )

        if file_path:
            # Save the image
            success = self.export_manager.save_image(current_image, file_path)

            if success:
                self.statusBar.showMessage(f"Saved current frame to {file_path}", 5000)

                # Update default directory
                self.config['export']['default_directory'] = os.path.dirname(file_path)
            else:
                self.statusBar.showMessage("Failed to save current frame", 5000)
                QMessageBox.critical(self, "Error", f"Failed to save current frame to {file_path}")

    def show_export_dialog(self):
        """Show export dialog."""
        if not self.fluorescence_stack and not self.mask_stack:
            QMessageBox.warning(self, "Warning", "No data to export")
            return

        dialog = ExportDialog(self)
        if dialog.exec():
            # Handle export options
            options = dialog.get_export_options()
            self.export_data(options)

    def export_data(self, options):
        """Export data according to options."""
        self.logger.info(f"Exporting data with options: {options}")
        self.statusBar.showMessage("Exporting data...")

        # Determine what to export
        if options['type'] == 'frames':
            # Export frames as images
            success = self.export_manager.export_frames(
                self.fluorescence_stack,
                self.mask_stack,
                options['output_dir'],
                options['format'],
                options['frames'],
                options['apply_overlay'],
                self.display_settings
            )
        elif options['type'] == 'movie':
            # Export as movie
            success = self.export_manager.export_movie(
                self.fluorescence_stack,
                self.mask_stack,
                options['output_file'],
                options['fps'],
                options['apply_overlay'],
                self.display_settings
            )
        elif options['type'] == 'data':
            # Export numerical data
            success = self.export_manager.export_data(
                self.fluorescence_stack,
                self.mask_stack,
                options['output_file'],
                options['include_stats'],
                options['roi_data'],
                self.rois
            )
        else:
            self.logger.error(f"Unknown export type: {options['type']}")
            success = False

        if success:
            self.statusBar.showMessage("Export completed successfully", 5000)
        else:
            self.statusBar.showMessage("Export failed", 5000)
            QMessageBox.critical(self, "Error", "Failed to export data")

    # Frame navigation
    def update_frame(self, frame_idx):
        """Update current frame."""
        self.logger.debug(f"Updating to frame {frame_idx}")

        # Ensure frame index is valid
        max_frames = 0
        if self.fluorescence_stack:
            max_frames = max(max_frames, self.fluorescence_stack.max_frames)
        if self.mask_stack:
            max_frames = max(max_frames, self.mask_stack.max_frames)

        if max_frames == 0:
            return

        frame_idx = max(0, min(frame_idx, max_frames - 1))

        # Update current frame
        self.current_frame = frame_idx

        # Get current frames based on visibility settings
        fluor_frame = None
        mask_frame = None

        if self.display_settings['fluorescence_visible'] and self.fluorescence_stack:
            fluor_frame = self.fluorescence_stack.get_frame(self.current_frame)

        if self.display_settings['mask_visible'] and self.mask_stack:
            mask_frame = self.mask_stack.get_frame(self.current_frame)

        # Update image view
        self.image_view.update_image(
            fluor_frame,
            mask_frame,
            self.display_settings['overlay_alpha'],
            self.display_settings['mask_color'],
            dict(self.display_settings)
        )

        # Update histogram if we have a visible fluorescence frame
        if fluor_frame is not None and self.display_settings['fluorescence_visible']:
            self.histogram.update_histogram(fluor_frame)

        # Update status bar
        self.statusBar.showMessage(f"Frame {frame_idx + 1}/{max_frames}", 3000)

    # Display functions
    def update_levels(self, levels):
        """Update display levels."""
        if levels == (self.display_settings.get('min_level'), self.display_settings.get('max_level')):
            return  # Avoid unnecessary updates
        self.display_settings['min_level'] = levels[0]
        self.display_settings['max_level'] = levels[1]
        self.update_display()

    def update_display_settings(self, settings):
        """Update display settings and refresh display."""
        # Check if settings actually changed
        changed = False
        for key, value in settings.items():
            if key not in self.display_settings or self.display_settings[key] != value:
                changed = True
                break

        if changed:
            self.display_settings.update(settings)
            self.update_display()

    def on_fluorescence_loaded(self, stack):
        """Handle fluorescence stack loading."""
        self.update_display()
        self.histogram.update_histogram(stack.get_frame(0))
        self.navigation_bar.update_frame_count(stack)
        self.timeline.update_timeline(stack)

    def on_mask_loaded(self, stack):
        """Handle mask stack loading."""
        self.update_display()

    def on_display_settings_changed(self, settings):
        """Handle changes to display settings."""
        self.logger.debug(f"Display settings changed: {settings}")
        self.display_settings.update(settings)
        self.update_display()

    def update_display(self):
        """Update the image display."""
        self.logger.debug("Updating display")

        # Get current frames based on visibility settings
        fluor_frame = None
        if self.display_settings['fluorescence_visible'] and self.fluorescence_stack:
            fluor_frame = self.fluorescence_stack.get_frame(self.current_frame)

        mask_frame = None
        if self.display_settings['mask_visible'] and self.mask_stack:
            mask_frame = self.mask_stack.get_frame(self.current_frame)

        # Update image view with all display settings
        self.image_view.update_image(
            fluor_frame,
            mask_frame,
            self.display_settings['overlay_alpha'],
            self.display_settings['mask_color'],
            dict(self.display_settings)  # Pass a copy to prevent recursion
        )

        # Update histogram if we have a visible fluorescence frame
        if fluor_frame is not None and self.display_settings['fluorescence_visible']:
            self.histogram.update_histogram(fluor_frame)

    def zoom_in(self):
        """Zoom in the image view."""
        self.image_view.zoom_in()
        self.display_settings['zoom_level'] = self.image_view.get_zoom_level()

    def zoom_out(self):
        """Zoom out the image view."""
        self.image_view.zoom_out()
        self.display_settings['zoom_level'] = self.image_view.get_zoom_level()

    def reset_zoom(self):
        """Reset zoom to original size."""
        self.image_view.reset_zoom()
        self.display_settings['zoom_level'] = 1.0

    # Dialog functions
    def show_settings_dialog(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            # Update config with new settings
            new_config = dialog.get_settings()
            self.config.update(new_config)

            # Apply settings
            self.apply_settings()

    def apply_settings(self):
        """Apply settings from config."""
        # Update display settings
        self.display_settings['mask_color'] = self.config['display']['default_mask_color']
        self.display_settings['overlay_alpha'] = self.config['display']['default_overlay_alpha']
        self.display_settings['auto_contrast'] = self.config['display']['auto_contrast']
        self.display_settings['colormap'] = self.config['display']['colormap']

        # Update navigation settings
        self.navigation_bar.set_frame_rate(self.config['navigation']['frame_rate'])
        self.navigation_bar.set_loop_playback(self.config['navigation']['loop_playback'])

        # Update image view
        if self.config['display']['show_scale_bar']:
            self.image_view.set_scale_bar(
                self.config['display']['scale_bar_size'],
                self.config['display']['scale_bar_units'],
                self.config['display']['pixels_per_unit']
            )
        else:
            self.image_view.hide_scale_bar()

        # Emit signal to update display
        self.display_settings_changed.emit(self.display_settings)

    def show_contrast_dialog(self):
        """Show contrast adjustment dialog."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        dialog = ContrastDialog(self.fluorescence_stack, self)
        if dialog.exec():
            # Apply contrast adjustment
            params = dialog.get_contrast_params()

            success = self.fluorescence_stack.apply_contrast(
                params['p_low'], params['p_high']
            )

            if success:
                self.statusBar.showMessage("Contrast adjustment applied", 5000)
                self.update_display()
            else:
                self.statusBar.showMessage("Failed to apply contrast adjustment", 5000)

    def show_projection_dialog(self):
        """Show Z-projection dialog."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        dialog = ProjectionDialog(self)
        if dialog.exec():
            # Create projection
            params = dialog.get_projection_params()

            projection = self.fluorescence_stack.create_z_projection(params['method'])

            if projection is not None:
                # Display projection
                self.image_view.show_projection(projection, params['method'])
                self.statusBar.showMessage(f"{params['method']} projection created", 5000)

                # Save if requested
                if params['save']:
                    self.export_manager.save_image(projection, params['save_path'])
            else:
                self.statusBar.showMessage("Failed to create projection", 5000)

    def show_registration_dialog(self):
        """Show registration dialog."""
        if not self.fluorescence_stack and not self.mask_stack:
            QMessageBox.warning(self, "Warning", "No data loaded for registration")
            return

        dialog = RegistrationDialog(self)
        if dialog.exec():
            # Apply registration
            params = dialog.get_registration_params()

            self.statusBar.showMessage("Performing image registration...")
            success = self.perform_registration(params)

            if success:
                self.statusBar.showMessage("Registration completed successfully", 5000)
                self.update_display()
            else:
                self.statusBar.showMessage("Registration failed", 5000)

    def perform_registration(self, params):
        """Perform image registration according to parameters."""
        self.logger.info(f"Performing registration with params: {params}")

        try:
            if params['stack'] == 'mask' and self.mask_stack and self.fluorescence_stack:
                # Register mask to fluorescence
                reference_frame = self.fluorescence_stack.get_frame(params['reference_frame'])

                for i in range(self.mask_stack.max_frames):
                    moving_frame = self.mask_stack.get_frame(i)

                    registered, transformation = self.image_processor.register_images(
                        reference_frame, moving_frame, params['method'],
                        upsample_factor=params['precision']
                    )

                    if registered is not None:
                        self.mask_stack.data[i] = registered

                return True

            elif params['stack'] == 'fluorescence' and self.fluorescence_stack and self.mask_stack:
                # Register fluorescence to mask
                reference_frame = self.mask_stack.get_frame(params['reference_frame'])

                for i in range(self.fluorescence_stack.max_frames):
                    moving_frame = self.fluorescence_stack.get_frame(i)

                    registered, transformation = self.image_processor.register_images(
                        reference_frame, moving_frame, params['method'],
                        upsample_factor=params['precision']
                    )

                    if registered is not None:
                        self.fluorescence_stack.data[i] = registered

                return True

            elif params['stack'] == 'self' and self.fluorescence_stack:
                # Register fluorescence stack to reference frame
                reference_frame = self.fluorescence_stack.get_frame(params['reference_frame'])

                for i in range(self.fluorescence_stack.max_frames):
                    if i == params['reference_frame']:
                        continue  # Skip reference frame

                    moving_frame = self.fluorescence_stack.get_frame(i)

                    registered, transformation = self.image_processor.register_images(
                        reference_frame, moving_frame, params['method'],
                        upsample_factor=params['precision']
                    )

                    if registered is not None:
                        self.fluorescence_stack.data[i] = registered

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in registration: {e}")
            return False

    def show_intensity_analysis_dialog(self):
        """Show intensity analysis dialog."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        dialog = AnalysisDialog(self, analysis_type='intensity')
        if dialog.exec():
            # Perform intensity analysis
            params = dialog.get_analysis_params()

            if params['roi_type'] == 'current':
                # Use current ROIs
                if not self.rois:
                    QMessageBox.warning(self, "Warning", "No ROIs defined")
                    return

                regions = list(self.rois.values())
            elif params['roi_type'] == 'mask' and self.mask_stack:
                # Use mask as ROI
                regions = None
                mask = self.mask_stack.get_frame(self.current_frame)
            else:
                # Use whole image
                regions = None
                mask = None

            # Perform analysis
            frame = self.fluorescence_stack.get_frame(self.current_frame)

            if regions:
                results = self.image_processor.measure_intensity(frame, regions=regions)
            elif mask is not None:
                results = self.image_processor.measure_intensity(frame, mask=mask)
            else:
                results = self.image_processor.measure_intensity(frame)

            # Send results to analysis completed handler
            analysis_data = {
                'type': 'intensity',
                'frame': self.current_frame,
                'results': results
            }
            self.analysis_completed.emit(analysis_data)

    def show_time_series_dialog(self):
        """Show time series analysis dialog."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        dialog = AnalysisDialog(self, analysis_type='time_series')
        if dialog.exec():
            # Perform time series analysis
            params = dialog.get_analysis_params()

            if params['roi_type'] == 'current':
                # Use current ROIs
                if not self.rois:
                    QMessageBox.warning(self, "Warning", "No ROIs defined")
                    return

                regions = list(self.rois.values())
                mask = None
            elif params['roi_type'] == 'mask' and self.mask_stack:
                # Use mask as ROI
                regions = None
                mask = self.mask_stack.get_frame(self.current_frame)
            else:
                # Use whole image
                regions = None
                mask = None

            # Perform analysis
            results = self.image_processor.extract_time_series(
                self.fluorescence_stack, mask=mask, regions=regions
            )

            # Send results to analysis completed handler
            analysis_data = {
                'type': 'time_series',
                'results': results
            }
            self.analysis_completed.emit(analysis_data)

    def show_feature_detection_dialog(self):
        """Show feature detection dialog."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        dialog = AnalysisDialog(self, analysis_type='feature')
        if dialog.exec():
            # Perform feature detection
            params = dialog.get_analysis_params()

            frame = self.fluorescence_stack.get_frame(self.current_frame)

            # Detect features
            features = self.image_processor.detect_features(
                frame, method=params['method'],
                min_sigma=params['min_size'],
                max_sigma=params['max_size'],
                threshold=params['threshold']
            )

            # Show detected features on image
            self.image_view.show_detected_features(features, params['method'])

            # Send results to analysis completed handler
            analysis_data = {
                'type': 'feature',
                'method': params['method'],
                'frame': self.current_frame,
                'features': features
            }
            self.analysis_completed.emit(analysis_data)

    def show_log_dialog(self):
        """Show log dialog."""
        from gui.dialogs import LogViewerDialog

        dialog = LogViewerDialog(self)
        dialog.exec()

    def show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Advanced TIFF Stack Viewer",
            "Advanced TIFF Stack Viewer v1.0\n\n"
            "A tool for visualizing and analyzing microscopy TIFF stacks.\n\n"
            "Developed for scientific image analysis.\n"
        )

    def show_documentation(self):
        """Show documentation."""
        # In a real application, this would open a browser or help window
        QMessageBox.information(
            self,
            "Documentation",
            "Please refer to the README.md file for documentation."
        )

    # ROI handling
    def handle_roi_created(self, roi_data):
      """Handle ROI creation."""
      self.logger.debug(f"Handling ROI creation with data: {roi_data}")

      try:
          # Store ROI data
          self.rois[roi_data['id']] = roi_data
          self.logger.debug(f"Stored ROI data for {roi_data['id']}")

          # Add to ROI list in toolbar
          self.roi_toolbar.add_roi(roi_data)
          self.logger.debug(f"Added ROI {roi_data['id']} to toolbar list")

          # Update status bar
          self.statusBar.showMessage(f"ROI created: {roi_data['id']}", 3000)

      except Exception as e:
          self.logger.error(f"Error handling ROI creation: {str(e)}")
          import traceback
          self.logger.error(traceback.format_exc())

    def handle_roi_modified(self, roi_data):
        """Handle ROI modification."""
        self.logger.debug(f"Handling ROI modification with data: {roi_data}")

        try:
            # Update stored ROI data
            self.rois[roi_data['id']] = roi_data
            self.logger.debug(f"Updated stored data for ROI {roi_data['id']}")

            # Update ROI in toolbar list
            self.roi_toolbar.update_roi(roi_data)
            self.logger.debug(f"Updated ROI {roi_data['id']} in toolbar list")

        except Exception as e:
            self.logger.error(f"Error handling ROI modification: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def handle_roi_deleted(self, roi_id):
        """Handle ROI deletion."""
        self.logger.debug(f"Handling ROI deletion for: {roi_id}")

        try:
            # Remove from stored ROIs
            if roi_id in self.rois:
                del self.rois[roi_id]
                self.logger.debug(f"Removed ROI {roi_id} from stored ROIs")

            # Remove from ROI list in toolbar
            self.roi_toolbar.remove_roi(roi_id)
            self.logger.debug(f"Removed ROI {roi_id} from toolbar list")

            # Update status bar
            self.statusBar.showMessage(f"ROI deleted: {roi_id}", 3000)

        except Exception as e:
            self.logger.error(f"Error handling ROI deletion: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def select_roi(self, roi_id):
        """Handle ROI selection."""
        self.logger.debug(f"Selecting ROI: {roi_id}")
        if roi_id in self.rois:
            # Highlight the ROI in the view
            self.image_view.select_roi(roi_id)
            self.statusBar.showMessage(f"Selected ROI: {roi_id}", 3000)

    def delete_roi(self, roi_id):
        """Delete a specific ROI."""
        self.logger.debug(f"Deleting ROI: {roi_id}")
        # First remove from image view
        self.image_view.remove_roi(roi_id)
        # Then remove from our records
        if roi_id in self.rois:
            del self.rois[roi_id]
        # Remove from toolbar list
        self.roi_toolbar.remove_roi(roi_id)
        # Update status
        self.statusBar.showMessage(f"Deleted ROI: {roi_id}", 3000)

    def clear_rois(self):
        """Clear all ROIs."""
        self.logger.debug("Clearing all ROIs")
        # Clear ROIs from image view
        self.image_view.clear_rois()
        # Clear our records
        self.rois.clear()
        # Clear toolbar list
        self.roi_toolbar.clear_rois()
        # Update status
        self.statusBar.showMessage("Cleared all ROIs", 3000)

    def analyze_roi(self, roi_id):
        """Analyze the selected ROI."""
        self.logger.debug(f"Analyzing ROI: {roi_id}")
        if roi_id not in self.rois:
            self.logger.warning(f"ROI {roi_id} not found for analysis")
            return

        roi_data = self.rois[roi_id]
        try:
            # Get current frame data
            if not self.fluorescence_stack:
                raise ValueError("No fluorescence data loaded")

            frame = self.fluorescence_stack.get_frame(self.current_frame)
            if frame is None:
                raise ValueError("Could not get current frame")

            # Get ROI mask based on type and points
            mask = np.zeros(frame.shape, dtype=bool)
            if roi_data['type'] == 'rectangle':
                y1, x1, y2, x2 = roi_data['points']
                mask[y1:y2, x1:x2] = True
            elif roi_data['type'] == 'ellipse':
                cy, cx, ry, rx = roi_data['points']
                y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
                mask[((x - cx)/rx)**2 + ((y - cy)/ry)**2 <= 1] = True
            elif roi_data['type'] == 'polygon':
                points = np.array(roi_data['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)

            # Calculate statistics
            stats = {
                'roi_id': roi_id,
                'frame': self.current_frame + 1,  # 1-based frame number for display
                'mean': float(np.mean(frame[mask])),
                'std': float(np.std(frame[mask])),
                'min': float(np.min(frame[mask])),
                'max': float(np.max(frame[mask])),
                'sum': float(np.sum(frame[mask])),
                'area': int(np.sum(mask))
            }

            # Show results dialog
            self.show_roi_analysis(stats)

        except Exception as e:
            self.logger.error(f"Error analyzing ROI: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to analyze ROI: {str(e)}")

    def show_roi_analysis(self, stats):
        """Show ROI analysis results."""
        title = "ROI Analysis Results"
        subtitle = f"ROI: {stats['roi_id']}, Frame: {stats['frame']}"
        dialog = ResultsDialog(title, subtitle, stats, self)
        dialog.exec()

    # Image processing
    def apply_filter(self, filter_type):
        """Apply filter to current fluorescence frame."""
        if not self.fluorescence_stack:
            QMessageBox.warning(self, "Warning", "No fluorescence data loaded")
            return

        self.logger.info(f"Applying {filter_type} filter")
        self.statusBar.showMessage(f"Applying {filter_type} filter...")

        # Get current frame
        frame = self.fluorescence_stack.get_frame(self.current_frame)

        # Apply filter
        filtered = self.image_processor.apply_filter(frame, filter_type)

        # Update frame in stack
        self.fluorescence_stack.data[self.current_frame] = filtered

        # Update display
        self.update_display()

        self.statusBar.showMessage(f"Applied {filter_type} filter to frame {self.current_frame+1}", 5000)

    # Analysis handling
    def handle_analysis_completed(self, analysis_data):
        """Handle completed analysis."""
        self.logger.info(f"Analysis completed: {analysis_data['type']}")

        if analysis_data['type'] == 'intensity':
            # Show intensity results
            self.show_intensity_results(analysis_data['results'], analysis_data['frame'])
        elif analysis_data['type'] == 'time_series':
            # Show time series results
            self.show_time_series_results(analysis_data['results'])
        elif analysis_data['type'] == 'feature':
            # Features already shown on image, just update status
            num_features = len(analysis_data['features'])
            self.statusBar.showMessage(
                f"Detected {num_features} features using {analysis_data['method']} method", 5000
            )

    def show_intensity_results(self, results, frame):
        """Show intensity analysis results."""
        from gui.dialogs import ResultsDialog

        dialog = ResultsDialog(
            "Intensity Analysis Results",
            f"Frame {frame+1}",
            results,
            self
        )
        dialog.exec()

    def show_time_series_results(self, results):
        """Show time series analysis results."""
        from gui.dialogs import TimeSeriesDialog

        dialog = TimeSeriesDialog(results, self)
        dialog.exec()

"""
ROI (Region of Interest) management toolbar.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QComboBox,
                           QLabel, QListWidget, QListWidgetItem, QHBoxLayout,
                           QToolButton, QSizePolicy, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
import logging


class ROIToolbar(QWidget):
    """Widget for managing Regions of Interest (ROIs)."""

    # Signals
    roi_type_changed = pyqtSignal(str)
    roi_selected = pyqtSignal(str)
    roi_delete_requested = pyqtSignal(str)
    roi_clear_requested = pyqtSignal()
    roi_analysis_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize ROI toolbar."""
        super().__init__(parent)

        # Initialize logger
        self.logger = logging.getLogger('tiff_stack_viewer')

        # Initialize variables
        self.current_roi_type = 'rectangle'

        # Set up UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # ROI creation controls
        creation_group = QGroupBox("Create ROI")
        creation_layout = QVBoxLayout()

        # ROI type selection
        creation_layout.addWidget(QLabel("ROI Type:"))
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItems(["Rectangle", "Ellipse", "Polygon"])
        self.roi_type_combo.currentTextChanged.connect(self.on_roi_type_changed)
        creation_layout.addWidget(self.roi_type_combo)

        # Creation instructions
        instruction_label = QLabel("Double click on image to create ROI")
        instruction_label.setWordWrap(True)
        creation_layout.addWidget(instruction_label)

        creation_group.setLayout(creation_layout)
        layout.addWidget(creation_group)

        # ROI list
        list_group = QGroupBox("ROI List")
        list_layout = QVBoxLayout()

        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self.on_roi_selected)
        list_layout.addWidget(self.roi_list)

        # ROI buttons
        buttons_layout = QHBoxLayout()

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        self.delete_button.setEnabled(False)  # Disabled until ROI is selected
        buttons_layout.addWidget(self.delete_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.on_clear_clicked)
        buttons_layout.addWidget(self.clear_button)

        list_layout.addLayout(buttons_layout)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # Analysis controls
        analysis_group = QGroupBox("ROI Analysis")
        analysis_layout = QVBoxLayout()

        self.analyze_button = QPushButton("Analyze Selected ROI")
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        self.analyze_button.setEnabled(False)  # Disabled until ROI is selected
        analysis_layout.addWidget(self.analyze_button)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Add stretch to bottom
        layout.addStretch()

    def add_roi(self, roi_data):
        """Add a new ROI to the list."""
        self.logger.debug(f"Adding ROI to list: {roi_data}")
        item = QListWidgetItem(f"{roi_data['id']} ({roi_data['type']})")
        item.setData(Qt.ItemDataRole.UserRole, roi_data['id'])
        self.roi_list.addItem(item)

    def update_roi(self, roi_data):
        """Update an existing ROI in the list."""
        # Find and update the item
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == roi_data['id']:
                item.setText(f"{roi_data['id']} ({roi_data['type']})")
                break

    def remove_roi(self, roi_id):
        """Remove an ROI from the list."""
        # Find and remove the item
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == roi_id:
                self.roi_list.takeItem(i)
                break

    def clear_rois(self):
        """Clear all ROIs from the list."""
        self.roi_list.clear()

    def on_roi_type_changed(self, type_text):
        """Handle ROI type selection changes."""
        roi_type = type_text.lower()
        self.current_roi_type = roi_type
        self.roi_type_changed.emit(roi_type)

    def on_roi_selected(self, item):
        """Handle ROI selection in the list."""
        self.logger.debug("ROI selected in list")
        if item is not None:
            roi_id = item.data(Qt.ItemDataRole.UserRole)
            self.delete_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.roi_selected.emit(roi_id)
        else:
            self.delete_button.setEnabled(False)
            self.analyze_button.setEnabled(False)

    def on_delete_clicked(self):
        """Handle delete button click."""
        selected_items = self.roi_list.selectedItems()
        if not selected_items:
            return

        roi_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.logger.debug(f"Delete clicked for ROI: {roi_id}")
        self.roi_delete_requested.emit(roi_id)

        # Clear selection and disable buttons
        self.roi_list.clearSelection()
        self.delete_button.setEnabled(False)
        self.analyze_button.setEnabled(False)

    def on_clear_clicked(self):
        """Handle clear all button click."""
        self.logger.debug("Clear all ROIs requested")
        self.roi_clear_requested.emit()

        # Clear list and disable buttons
        self.roi_list.clear()
        self.delete_button.setEnabled(False)
        self.analyze_button.setEnabled(False)

    def on_analyze_clicked(self):
        """Handle analyze button click."""
        selected_items = self.roi_list.selectedItems()
        if not selected_items:
            return

        roi_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.logger.debug(f"Analyze clicked for ROI: {roi_id}")
        self.roi_analysis_requested.emit(roi_id)

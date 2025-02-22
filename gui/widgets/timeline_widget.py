"""
Timeline widget for visualizing and navigating TIFF stacks.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QLinearGradient

import pyqtgraph as pg


class TimelineWidget(QWidget):
    """Widget for displaying and navigating through time series data."""
    
    # Signals
    frame_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        """Initialize timeline widget."""
        super().__init__(parent)
        
        # Initialize variables
        self.current_frame = 0
        self.max_frames = 0
        self.intensity_data = None
        
        # Set up UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setTracking(True)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(self.frame_slider)
        
        # Create intensity plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)
        self.plot_widget.setMaximumHeight(100)
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.hideButtons()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot line
        self.plot_line = self.plot_widget.plot([], [], pen=pg.mkPen(color='b', width=2))
        
        # Add vertical line for current frame
        self.current_frame_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color='r', width=1))
        self.plot_widget.addItem(self.current_frame_line)
        
        # Connect mouse click on plot
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        
        layout.addWidget(self.plot_widget)
    
    def update_timeline(self, image_stack=None):
        """Update timeline display with new image stack data."""
        if image_stack is None:
            return
        
        # Update frame count
        self.max_frames = image_stack.max_frames
        self.frame_slider.setRange(0, max(0, self.max_frames - 1))
        
        # Calculate mean intensity for each frame
        if self.max_frames > 0:
            intensity_data = []
            for i in range(self.max_frames):
                frame = image_stack.get_frame(i)
                if frame is not None:
                    mean_intensity = np.mean(frame)
                    intensity_data.append(mean_intensity)
                else:
                    intensity_data.append(0)
            
            self.intensity_data = intensity_data
            
            # Update plot
            self.plot_line.setData(range(len(intensity_data)), intensity_data)
            
            # Set y range to min/max of data with 10% padding
            y_min = min(intensity_data)
            y_max = max(intensity_data)
            y_range = y_max - y_min
            self.plot_widget.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Set x range to frame count
            self.plot_widget.setXRange(0, len(intensity_data) - 1)
    
    def set_current_frame(self, frame):
        """Set the current frame."""
        # Ensure frame is within valid range
        frame = max(0, min(frame, self.max_frames - 1)) if self.max_frames > 0 else 0
        
        # Update current frame
        self.current_frame = frame
        
        # Update UI
        self.frame_slider.setValue(frame)
        self.current_frame_line.setValue(frame)
        
        # Emit signal
        self.frame_changed.emit(frame)
    
    def on_slider_value_changed(self, value):
        """Handle frame slider value changes."""
        # Only update if different from current frame
        if value != self.current_frame:
            self.set_current_frame(value)
    
    def on_plot_clicked(self, event):
        """Handle mouse clicks on the plot."""
        # Get mouse position in plot coordinates
        pos = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        
        # Round to nearest frame index
        frame = int(round(pos.x()))
        
        # Set current frame if within valid range
        if 0 <= frame < self.max_frames:
            self.set_current_frame(frame)

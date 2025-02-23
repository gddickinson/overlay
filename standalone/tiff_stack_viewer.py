import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog,
                            QComboBox, QCheckBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from tifffile import imread

class TiffStackViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.fluorescence_data = None
        self.mask_data = None
        self.current_frame = 0
        self.overlay_alpha = 0.5
        self.mask_color = [255, 0, 0]  # Default red for mask
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('TIFF Stack Viewer')
        self.setGeometry(100, 100, 1000, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create pyqtgraph ImageView for displaying images
        self.image_view = pg.ImageView()
        main_layout.addWidget(self.image_view)
        
        # Create bottom controls panel
        controls_layout = QHBoxLayout()
        
        # Frame navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QGridLayout()
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_frame)
        
        self.frame_label = QLabel("Frame: 0/0")
        
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.previous_frame)
        
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_frame)
        
        nav_layout.addWidget(QLabel("Frame:"), 0, 0)
        nav_layout.addWidget(self.frame_slider, 0, 1, 1, 3)
        nav_layout.addWidget(self.frame_label, 0, 4)
        nav_layout.addWidget(prev_button, 1, 1)
        nav_layout.addWidget(next_button, 1, 3)
        nav_group.setLayout(nav_layout)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout()
        
        # Opacity control
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        
        # Color selection for mask
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"])
        self.color_combo.currentIndexChanged.connect(self.update_mask_color)
        
        # Visibility toggles
        self.show_fluorescence = QCheckBox("Show Fluorescence")
        self.show_fluorescence.setChecked(True)
        self.show_fluorescence.stateChanged.connect(self.update_display)
        
        self.show_mask = QCheckBox("Show Mask")
        self.show_mask.setChecked(True)
        self.show_mask.stateChanged.connect(self.update_display)
        
        display_layout.addWidget(QLabel("Overlay Opacity:"), 0, 0)
        display_layout.addWidget(self.alpha_slider, 0, 1)
        display_layout.addWidget(QLabel("Mask Color:"), 1, 0)
        display_layout.addWidget(self.color_combo, 1, 1)
        display_layout.addWidget(self.show_fluorescence, 2, 0)
        display_layout.addWidget(self.show_mask, 2, 1)
        display_group.setLayout(display_layout)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QGridLayout()
        
        load_fluorescence_btn = QPushButton("Load Fluorescence Stack")
        load_fluorescence_btn.clicked.connect(self.load_fluorescence)
        
        load_mask_btn = QPushButton("Load Mask Stack")
        load_mask_btn.clicked.connect(self.load_mask)
        
        file_layout.addWidget(load_fluorescence_btn, 0, 0)
        file_layout.addWidget(load_mask_btn, 0, 1)
        file_group.setLayout(file_layout)
        
        # Add all control groups to the control layout
        controls_layout.addWidget(nav_group)
        controls_layout.addWidget(display_group)
        controls_layout.addWidget(file_group)
        
        main_layout.addLayout(controls_layout)
        
        self.setCentralWidget(central_widget)
        
    def load_fluorescence(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Fluorescence TIFF Stack", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.fluorescence_data = imread(file_path)
                print(f"Loaded fluorescence data shape: {self.fluorescence_data.shape}")
                
                if len(self.fluorescence_data.shape) == 2:
                    # Single frame - reshape to add dimension
                    self.fluorescence_data = self.fluorescence_data.reshape(1, *self.fluorescence_data.shape)
                
                # Update slider range
                self.update_slider_range()
                self.update_display()
                
            except Exception as e:
                print(f"Error loading fluorescence data: {e}")
    
    def load_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Mask TIFF Stack", "", "TIFF Files (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.mask_data = imread(file_path)
                print(f"Loaded mask data shape: {self.mask_data.shape}")
                
                if len(self.mask_data.shape) == 2:
                    # Single frame - reshape to add dimension
                    self.mask_data = self.mask_data.reshape(1, *self.mask_data.shape)
                
                # Update slider range
                self.update_slider_range()
                self.update_display()
                
            except Exception as e:
                print(f"Error loading mask data: {e}")
    
    def update_slider_range(self):
        max_frames = 0
        
        if self.fluorescence_data is not None:
            max_frames = max(max_frames, self.fluorescence_data.shape[0])
        
        if self.mask_data is not None:
            max_frames = max(max_frames, self.mask_data.shape[0])
        
        if max_frames > 0:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
        else:
            self.frame_slider.setEnabled(False)
    
    def update_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.frame_label.setText(f"Frame: {frame_idx+1}/{self.frame_slider.maximum()+1}")
        self.update_display()
    
    def previous_frame(self):
        if self.frame_slider.isEnabled() and self.current_frame > 0:
            self.frame_slider.setValue(self.current_frame - 1)
    
    def next_frame(self):
        if self.frame_slider.isEnabled() and self.current_frame < self.frame_slider.maximum():
            self.frame_slider.setValue(self.current_frame + 1)
    
    def update_alpha(self, value):
        self.overlay_alpha = value / 100.0
        self.update_display()
    
    def update_mask_color(self, index):
        colors = {
            0: [255, 0, 0],    # Red
            1: [0, 255, 0],    # Green
            2: [0, 0, 255],    # Blue
            3: [255, 255, 0],  # Yellow
            4: [255, 0, 255],  # Magenta
            5: [0, 255, 255],  # Cyan
        }
        self.mask_color = colors[index]
        self.update_display()
    
    def update_display(self):
        if self.fluorescence_data is None and self.mask_data is None:
            return
        
        # Create a blank RGB image
        if self.fluorescence_data is not None:
            height, width = self.fluorescence_data.shape[1:3]
        else:
            height, width = self.mask_data.shape[1:3]
        
        display_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get current frame of fluorescence data if available and visible
        if self.fluorescence_data is not None and self.show_fluorescence.isChecked():
            fluor_frame_idx = min(self.current_frame, self.fluorescence_data.shape[0] - 1)
            fluor_frame = self.fluorescence_data[fluor_frame_idx]
            
            # Normalize to 0-255 for display
            if fluor_frame.dtype != np.uint8:
                fluor_frame = ((fluor_frame - fluor_frame.min()) / 
                            (fluor_frame.max() - fluor_frame.min() + 1e-6) * 255).astype(np.uint8)
            
            # Add to all channels (grayscale)
            for i in range(3):
                display_image[:, :, i] = fluor_frame
        
        # Get current frame of mask data if available and visible
        if self.mask_data is not None and self.show_mask.isChecked():
            mask_frame_idx = min(self.current_frame, self.mask_data.shape[0] - 1)
            mask_frame = self.mask_data[mask_frame_idx]
            
            # Binarize if not already binary
            binary_mask = mask_frame > 0
            
            # Apply colored mask with alpha blending
            for i in range(3):
                mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                mask_channel[binary_mask] = self.mask_color[i]
                
                # Alpha blend
                display_image[:, :, i] = (display_image[:, :, i] * (1 - self.overlay_alpha * binary_mask) + 
                                     mask_channel * self.overlay_alpha).astype(np.uint8)
        
        # Update the image display
        self.image_view.setImage(display_image.transpose(2, 0, 1), autoLevels=False)


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = TiffStackViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

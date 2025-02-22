"""
Navigation bar for TIFF stack browsing.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel,
                           QSpinBox, QCheckBox, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


class NavigationBar(QWidget):
    """Widget for frame navigation controls."""

    # Signals
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialize navigation bar."""
        super().__init__(parent)

        # Initialize variables
        self.current_frame = 0
        self.max_frames = 0
        self.playing = False
        self.frame_rate = 10
        self.loop_playback = True
        self.playback_direction = 1  # 1 for forward, -1 for backward

        # Create playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.goto_next)

        # Set up UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # First frame button
        self.first_button = QPushButton("⏮")
        self.first_button.setToolTip("Go to first frame")
        self.first_button.clicked.connect(self.goto_first)
        layout.addWidget(self.first_button)

        # Previous frame button
        self.prev_button = QPushButton("⏪")
        self.prev_button.setToolTip("Previous frame")
        self.prev_button.clicked.connect(self.goto_previous)
        layout.addWidget(self.prev_button)

        # Play/pause button
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Play/Pause")
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        # Next frame button
        self.next_button = QPushButton("⏩")
        self.next_button.setToolTip("Next frame")
        self.next_button.clicked.connect(self.goto_next)
        layout.addWidget(self.next_button)

        # Last frame button
        self.last_button = QPushButton("⏭")
        self.last_button.setToolTip("Go to last frame")
        self.last_button.clicked.connect(self.goto_last)
        layout.addWidget(self.last_button)

        # Frame counter
        layout.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setValue(1)
        self.frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        layout.addWidget(self.frame_spin)

        self.frame_count_label = QLabel("/ 1")
        layout.addWidget(self.frame_count_label)

        # Frame rate control
        layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(self.frame_rate)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        layout.addWidget(self.fps_spin)

        # Loop playback checkbox
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(self.loop_playback)
        self.loop_checkbox.stateChanged.connect(self.on_loop_changed)
        layout.addWidget(self.loop_checkbox)

        # Add stretch to push everything to the left
        layout.addStretch()

    def update_frame_count(self, image_stack=None):
        """Update total frame count."""
        if image_stack is not None:
            self.max_frames = image_stack.max_frames

        # Update UI
        self.frame_spin.setMaximum(max(1, self.max_frames))
        self.frame_count_label.setText(f"/ {self.max_frames}")

        # Update current frame if needed
        if self.current_frame >= self.max_frames:
            self.set_current_frame(0)

    def set_current_frame(self, frame):
        """Set the current frame."""
        # Ensure frame is within valid range
        frame = max(0, min(frame, self.max_frames - 1)) if self.max_frames > 0 else 0

        # Update current frame
        self.current_frame = frame

        # Update UI
        self.frame_spin.setValue(frame + 1)  # +1 for UI (zero-indexed internally, 1-indexed in UI)

        # Emit signal
        self.frame_changed.emit(frame)

    def goto_first(self):
        """Go to the first frame."""
        self.set_current_frame(0)

    def goto_previous(self):
        """Go to the previous frame."""
        self.set_current_frame(self.current_frame - 1)

    def goto_next(self):
        """Go to the next frame."""
        if self.playing:
            # Handle loop playback logic
            next_frame = self.current_frame + self.playback_direction

            # Check if we've reached the end or beginning
            if next_frame >= self.max_frames:
                if self.loop_playback:
                    next_frame = 0
                else:
                    self.stop_playback()
                    return
            elif next_frame < 0:
                if self.loop_playback:
                    next_frame = self.max_frames - 1
                else:
                    self.stop_playback()
                    return

            self.set_current_frame(next_frame)
        else:
            # Normal next frame
            self.set_current_frame(self.current_frame + 1)

    def goto_last(self):
        """Go to the last frame."""
        self.set_current_frame(self.max_frames - 1)

    def toggle_playback(self):
        """Toggle playback state."""
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start playback."""
        self.playing = True
        self.play_button.setText("⏸")
        self.playback_timer.start(1000 // self.frame_rate)

    def stop_playback(self):
        """Stop playback."""
        self.playing = False
        self.play_button.setText("▶")
        self.playback_timer.stop()

    def set_frame_rate(self, fps):
        """Set playback frame rate."""
        self.frame_rate = max(1, min(fps, 60))
        self.fps_spin.setValue(self.frame_rate)

        # Update timer interval if playing
        if self.playing:
            self.playback_timer.setInterval(1000 // self.frame_rate)

    def set_loop_playback(self, loop):
        """Set loop playback option."""
        self.loop_playback = loop
        self.loop_checkbox.setChecked(loop)

    def on_frame_spin_changed(self, value):
        """Handle frame spin box value changes."""
        # Convert from 1-indexed (UI) to 0-indexed (internal)
        frame = value - 1

        # Only update if different from current frame
        if frame != self.current_frame:
            self.set_current_frame(frame)

    def on_fps_changed(self, value):
        """Handle FPS spin box value changes."""
        self.set_frame_rate(value)

    def on_loop_changed(self, state):
        """Handle loop checkbox state changes."""
        self.loop_playback = (state == Qt.CheckState.Checked)

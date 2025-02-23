"""
Enhanced image viewing widget.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QAction, QCursor
import pyqtgraph as pg



class EnhancedImageView(QWidget):

    # Custom signals
    roi_created = pyqtSignal(dict)
    roi_modified = pyqtSignal(dict)
    roi_deleted = pyqtSignal(str)
    position_changed = pyqtSignal(float, float, float)  # x, y, intensity


    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize logger
        self.logger = logging.getLogger('tiff_stack_viewer')

        # Initialize variables
        self.current_image = None
        self.current_overlay = None
        self.overlay_alpha = 0.5
        self.overlay_color = [255, 0, 0]  # Default red
        self.zoom_level = 1.0
        self.rois = {}
        self.next_roi_id = 1
        self.current_roi_type = 'rectangle'
        self.projection_mode = False
        self.scale_bar_visible = False
        self.scale_bar_size = 50
        self.scale_bar_units = 'μm'
        self.pixels_per_unit = 1.0
        self._updating = False  # Flag to prevent recursive updates

        # Set up UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create ImageView widget from PyQtGraph
        self.view = pg.ImageView()
        self.view.ui.roiBtn.hide()  # Hide default ROI button
        self.view.ui.menuBtn.hide()  # Hide default menu button

        # Set default view settings
        self.view.getImageItem().setAutoDownsample(True)
        self.view.setLevels(0, 255)  # Set default levels

        # Connect signals
        self.view.scene.sigMouseMoved.connect(self.mouse_moved)
        self.view.scene.sigMouseClicked.connect(self.mouse_clicked)

        # Add custom context menu
        self.view.scene.contextMenu = None  # Disable default context menu
        self.view.scene.sigMouseClicked.connect(self.context_menu_event)

        # Add view to layout
        layout.addWidget(self.view)

        # Create scale bar
        self.scale_bar_item = pg.ScaleBar(size=self.scale_bar_size, suffix=self.scale_bar_units)
        self.scale_bar_item.setParentItem(self.view.getView())
        self.scale_bar_item.anchor((1, 1), (1, 1), offset=(-20, -20))
        self.scale_bar_item.hide()

        # Create labels for showing position and intensity
        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.info_label.setStyleSheet("background-color: rgba(0, 0, 0, 128); color: white; padding: 5px;")
        self.info_label.setVisible(False)

        # Set additional view options
        self.view.getView().setAspectLocked(True)
        self.view.getView().setLimits(maxXRange=10000, maxYRange=10000)

    def update_image(self, image=None, overlay=None, alpha=None, color=None, display_settings=None):
        """Update the displayed image."""
        if self._updating:  # Prevent recursive updates
            return

        self._updating = True
        try:
            self.logger.debug(f"Update image called with settings: {display_settings}")

            # Update stored values only if provided
            if image is not None:
                self.current_image = image
            if overlay is not None:
                self.current_overlay = overlay
            if alpha is not None:
                self.overlay_alpha = alpha
            if color is not None:
                self.overlay_color = color

            # Get display settings
            if display_settings is None:
                display_settings = {}

            # Clear view if nothing should be visible
            if not any([
                display_settings.get('fluorescence_visible', True),
                display_settings.get('mask_visible', True)
            ]):
                self.logger.debug("Nothing visible, clearing view")
                self.view.clear()
                return

            # Start with fluorescence image if visible
            display_image = None
            if display_settings.get('fluorescence_visible', True) and self.current_image is not None:
                self.logger.debug("Setting fluorescence image")
                if len(self.current_image.shape) == 2:
                    display_image = np.stack([self.current_image] * 3, axis=2)
                else:
                    display_image = self.current_image.copy()

            # Add mask overlay if visible
            if (display_settings.get('mask_visible', True) and
                self.current_overlay is not None and
                self.overlay_alpha > 0 and
                display_image is not None):
                self.logger.debug("Adding mask overlay")
                binary_mask = self.current_overlay > 0

                for i in range(3):
                    mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                    mask_channel[binary_mask] = self.overlay_color[i]
                    display_image[..., i] = (
                        display_image[..., i] * (1 - self.overlay_alpha * binary_mask) +
                        mask_channel * self.overlay_alpha
                    ).astype(np.uint8)

            # If we don't have an image to display, clear the view
            if display_image is None:
                self.logger.debug("No image to display, clearing view")
                self.view.clear()
                return

            # Update the view
            self.logger.debug(f"Setting final image with shape {display_image.shape}")
            if len(display_image.shape) == 3:
                self.view.setImage(
                    display_image.transpose(2, 0, 1),
                    autoLevels=display_settings.get('auto_contrast', False)
                )
            else:
                self.view.setImage(
                    display_image,
                    autoLevels=display_settings.get('auto_contrast', False)
                )

            # Apply colormap if needed
            if 'colormap' in display_settings and len(display_image.shape) == 2:
                lut = self._get_colormap_lut(display_settings['colormap'])
                if lut is not None:
                    self.view.getImageItem().setLookupTable(lut)

        except Exception as e:
            self.logger.error(f"Error in update_image: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

        finally:
            self._updating = False

    def _create_display_image(self):
        """Create the display image combining fluorescence and overlay."""
        if self.current_image is None:
            return None

        # Start with fluorescence image
        if len(self.current_image.shape) == 2:
            display_image = np.stack([self.current_image] * 3, axis=2)
        else:
            display_image = self.current_image.copy()

        # Add overlay if available
        if self.current_overlay is not None and self.overlay_alpha > 0:
            # Apply overlay with color
            binary_mask = self.current_overlay > 0

            for i in range(3):
                mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                mask_channel[binary_mask] = self.overlay_color[i]

                # Alpha blend
                display_image[..., i] = (
                    display_image[..., i] * (1 - self.overlay_alpha * binary_mask) +
                    mask_channel * self.overlay_alpha
                ).astype(np.uint8)

        return display_image

    def _get_display_levels(self, image, settings):
        """Get display levels based on settings."""
        if settings.get('auto_contrast', True):
            # Calculate percentiles for contrast
            flat_image = image.flatten()
            min_level = float(np.percentile(flat_image, 1))
            max_level = float(np.percentile(flat_image, 99))
        else:
            min_level = float(settings.get('min_level', 0))
            max_level = float(settings.get('max_level', 255))

        return (min_level, max_level)

    def _get_colormap_lut(self, colormap_name):
        """Get lookup table for colormap."""
        try:
            cmap = pg.colormap.get(colormap_name)
            return cmap.getLookupTable()
        except Exception:
            return None

    def get_current_image(self):
        """Get current displayed image."""
        return self.view.getImageItem().image

    def show_projection(self, projection, method='max'):
        """Show Z-projection."""
        self.projection_mode = True

        # Update view with projection
        self.view.setImage(projection, autoLevels=True)

        # Add label to indicate projection mode
        self.view.getView().setTitle(f"{method.capitalize()} Projection")

    def show_detected_features(self, features, method='blob'):
        """Show detected features on the image."""
        # Clear any existing feature overlays
        self.clear_feature_overlays()

        # Add features as scatter points
        if method == 'blob':
            # Features are blobs with (y, x, r) format
            positions = [(x, y) for y, x, _ in features]
            sizes = [2 * r for _, _, r in features]

            scatter = pg.ScatterPlotItem(
                pos=positions,
                size=sizes,
                pen='r',
                brush=(255, 0, 0, 100)
            )
            self.view.getView().addItem(scatter)

        elif method == 'corner':
            # Features are corner points with (y, x) format
            positions = [(x, y) for y, x in features]

            scatter = pg.ScatterPlotItem(
                pos=positions,
                size=10,
                pen='g',
                brush=(0, 255, 0, 100),
                symbol='+'
            )
            self.view.getView().addItem(scatter)

        # Store scatter item for later removal
        self.feature_overlay = scatter

    def clear_feature_overlays(self):
        """Clear feature overlays."""
        if hasattr(self, 'feature_overlay'):
            self.view.getView().removeItem(self.feature_overlay)
            del self.feature_overlay

    def mouse_moved(self, pos):
        """Handle mouse movement to show position and intensity."""
        # Convert position to image coordinates
        view_pos = self.view.getView().mapSceneToView(pos)
        x, y = int(view_pos.x()), int(view_pos.y())

        # Get image shape
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]

            # Check if position is within image bounds
            if 0 <= x < width and 0 <= y < height:
                # Get intensity at position
                if len(self.current_image.shape) == 2:
                    intensity = self.current_image[y, x]
                    self.info_label.setText(f"X: {x}, Y: {y}, Value: {intensity}")
                else:
                    r, g, b = self.current_image[y, x]
                    self.info_label.setText(f"X: {x}, Y: {y}, RGB: ({r}, {g}, {b})")

                # Show info label
                self.info_label.setVisible(True)

                # Emit signal
                self.position_changed.emit(x, y, intensity if len(self.current_image.shape) == 2 else 0)
            else:
                self.info_label.setVisible(False)

    def mouse_clicked(self, event):
        """Handle mouse clicks."""
        if event.button() == Qt.MouseButton.LeftButton and event.double():
            # Double click - start ROI creation
            pos = event.pos()
            view_pos = self.view.getView().mapSceneToView(pos)
            self.start_roi_creation(view_pos.x(), view_pos.y())

    def context_menu_event(self, event):
        """Show custom context menu on right click."""
        if event.button() == Qt.MouseButton.RightButton:
            # Create context menu
            menu = QMenu(self)

            # Add actions
            zoom_in_action = QAction("Zoom In", self)
            zoom_in_action.triggered.connect(self.zoom_in)
            menu.addAction(zoom_in_action)

            zoom_out_action = QAction("Zoom Out", self)
            zoom_out_action.triggered.connect(self.zoom_out)
            menu.addAction(zoom_out_action)

            reset_zoom_action = QAction("Reset Zoom", self)
            reset_zoom_action.triggered.connect(self.reset_zoom)
            menu.addAction(reset_zoom_action)

            menu.addSeparator()

            # ROI submenu
            roi_menu = QMenu("Create ROI", self)

            rect_roi_action = QAction("Rectangle", self)
            rect_roi_action.triggered.connect(lambda: self.set_roi_type('rectangle'))
            roi_menu.addAction(rect_roi_action)

            ellipse_roi_action = QAction("Ellipse", self)
            ellipse_roi_action.triggered.connect(lambda: self.set_roi_type('ellipse'))
            roi_menu.addAction(ellipse_roi_action)

            polygon_roi_action = QAction("Polygon", self)
            polygon_roi_action.triggered.connect(lambda: self.set_roi_type('polygon'))
            roi_menu.addAction(polygon_roi_action)

            menu.addMenu(roi_menu)

            # Clear ROIs action
            clear_rois_action = QAction("Clear All ROIs", self)
            clear_rois_action.triggered.connect(self.clear_rois)
            menu.addAction(clear_rois_action)

            menu.addSeparator()

            # Toggle histogram action
            toggle_hist_action = QAction("Toggle Histogram", self)
            toggle_hist_action.setCheckable(True)
            toggle_hist_action.setChecked(self.view.ui.histogram.isVisible())
            toggle_hist_action.triggered.connect(
                lambda checked: self.view.ui.histogram.setVisible(checked)
            )
            menu.addAction(toggle_hist_action)

            # Show menu
            menu.exec(QCursor.pos())

    def set_roi_type(self, roi_type):
        """Set the type of ROI to create."""
        self.current_roi_type = roi_type
        self.logger.debug(f"ROI type set to {roi_type}")

    def start_roi_creation(self, x, y):
        """Start creating a new ROI."""
        self.logger.debug(f"Starting ROI creation at ({x}, {y})")

        roi_id = f"roi_{self.next_roi_id}"
        self.next_roi_id += 1

        if self.current_roi_type == 'rectangle':
            # Create rectangular ROI
            roi = pg.RectROI([x, y], [100, 100], pen=(0, 255, 0))
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addTranslateHandle([0.5, 0.5])

            # Connect ROI signals
            roi.sigRegionChangeFinished.connect(lambda: self.roi_changed(roi_id))

            # Add to view
            self.view.getView().addItem(roi)

            # Store ROI
            self.rois[roi_id] = {
                'id': roi_id,
                'type': 'rectangle',
                'roi_object': roi,
                'label': pg.TextItem(roi_id, color=(0, 255, 0))
            }

            # Add label
            self.view.getView().addItem(self.rois[roi_id]['label'])
            self.update_roi_label(roi_id)

            # Emit signal
            roi_data = self.get_roi_data(roi_id)
            self.roi_created.emit(roi_data)

        elif self.current_roi_type == 'ellipse':
            # Create elliptical ROI
            roi = pg.EllipseROI([x, y], [100, 100], pen=(0, 255, 0))
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addTranslateHandle([0.5, 0.5])

            # Connect ROI signals
            roi.sigRegionChangeFinished.connect(lambda: self.roi_changed(roi_id))

            # Add to view
            self.view.getView().addItem(roi)

            # Store ROI
            self.rois[roi_id] = {
                'id': roi_id,
                'type': 'ellipse',
                'roi_object': roi,
                'label': pg.TextItem(roi_id, color=(0, 255, 0))
            }

            # Add label
            self.view.getView().addItem(self.rois[roi_id]['label'])
            self.update_roi_label(roi_id)

            # Emit signal
            roi_data = self.get_roi_data(roi_id)
            self.roi_created.emit(roi_data)

        elif self.current_roi_type == 'polygon':
            # Create polygon ROI
            roi = pg.PolyLineROI([[x, y], [x+100, y], [x+100, y+100], [x, y+100]], closed=True, pen=(0, 255, 0))

            # Connect ROI signals
            roi.sigRegionChangeFinished.connect(lambda: self.roi_changed(roi_id))

            # Add to view
            self.view.getView().addItem(roi)

            # Store ROI
            self.rois[roi_id] = {
                'id': roi_id,
                'type': 'polygon',
                'roi_object': roi,
                'label': pg.TextItem(roi_id, color=(0, 255, 0))
            }

            # Add label
            self.view.getView().addItem(self.rois[roi_id]['label'])
            self.update_roi_label(roi_id)

            # Emit signal
            roi_data = self.get_roi_data(roi_id)
            self.roi_created.emit(roi_data)

    def roi_changed(self, roi_id):
        """Handle ROI change."""
        self.logger.debug(f"ROI changed: {roi_id}")

        # Update label position
        self.update_roi_label(roi_id)

        # Emit signal with updated data
        roi_data = self.get_roi_data(roi_id)
        self.roi_modified.emit(roi_data)

    def update_roi_label(self, roi_id):
        """Update position of ROI label."""
        if roi_id in self.rois:
            roi = self.rois[roi_id]['roi_object']
            label = self.rois[roi_id]['label']

            # Get ROI position
            pos = roi.pos()

            # Move label to top-left of ROI
            label.setPos(pos[0], pos[1])

    def get_roi_data(self, roi_id):
        """Get data for ROI."""
        if roi_id in self.rois:
            roi_info = self.rois[roi_id]
            roi = roi_info['roi_object']

            # Get ROI position and size
            pos = roi.pos()
            size = roi.size()

            # Create ROI data dictionary
            roi_data = {
                'id': roi_id,
                'type': roi_info['type']
            }

            if roi_info['type'] == 'rectangle':
                # Rectangle data
                roi_data['points'] = [
                    int(pos[1]),              # y1
                    int(pos[0]),              # x1
                    int(pos[1] + size[1]),    # y2
                    int(pos[0] + size[0])     # x2
                ]

            elif roi_info['type'] == 'ellipse':
                # Ellipse data
                center_x = pos[0] + size[0]/2
                center_y = pos[1] + size[1]/2
                radius_x = size[0]/2
                radius_y = size[1]/2

                roi_data['points'] = [
                    int(center_y),     # center y
                    int(center_x),     # center x
                    int(radius_y),     # radius y
                    int(radius_x)      # radius x
                ]

            elif roi_info['type'] == 'polygon':
                # Polygon data - convert to absolute coordinates
                handles = roi.getState()['points']
                pos = roi.pos()

                points = []
                for h in handles:
                    # Add position offset to handle positions
                    points.append([int(pos[1] + h[1]), int(pos[0] + h[0])])

                roi_data['points'] = points

            return roi_data

        return None

    def remove_roi(self, roi_id):
        """Remove a specific ROI."""
        self.logger.debug(f"Removing ROI: {roi_id}")
        if roi_id in self.rois:
            # Remove ROI object from view
            self.view.getView().removeItem(self.rois[roi_id]['roi_object'])
            # Remove label
            if 'label' in self.rois[roi_id]:
                self.view.getView().removeItem(self.rois[roi_id]['label'])
            # Remove from our dictionary
            del self.rois[roi_id]
            self.roi_deleted.emit(roi_id)

    def clear_rois(self):
        """Clear all ROIs."""
        self.logger.debug("Clearing all ROIs")
        # Make a copy of the keys since we'll be modifying the dictionary
        roi_ids = list(self.rois.keys())
        for roi_id in roi_ids:
            self.remove_roi(roi_id)

        # Reset ROI counter
        self.next_roi_id = 1

    def select_roi(self, roi_id):
        """Select and highlight an ROI."""
        self.logger.debug(f"Selecting ROI: {roi_id}")
        # Reset all ROIs to normal state
        for roi_info in self.rois.values():
            roi_info['roi_object'].setPen(pg.mkPen('g'))  # Normal state: green

        # Highlight selected ROI
        if roi_id in self.rois:
            # Highlight in yellow
            self.rois[roi_id]['roi_object'].setPen(pg.mkPen('y', width=2))

    def zoom_in(self):
        """Zoom in the view."""
        self.zoom_level *= 1.25
        view_box = self.view.getView()
        view_box.scaleBy((1.25, 1.25))

    def zoom_out(self):
        """Zoom out the view."""
        self.zoom_

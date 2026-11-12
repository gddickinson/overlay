"""Pre-fill the processing/analysis dialogs with the values chosen in the controls panel.

The controls panel (gui/controls_panel.py) passes its current choices to the main window, which opens the matching
dialog with those values already selected, so the user confirms them instead of choosing twice.
"""

PROJECTION_RADIOS = {
    'max': 'max_radio', 'maximum': 'max_radio', 'min': 'min_radio', 'minimum': 'min_radio',
    'mean': 'mean_radio', 'median': 'median_radio', 'std': 'std_radio', 'standard deviation': 'std_radio',
}
ROI_RADIOS = {'current': 'current_roi_radio', 'mask': 'mask_roi_radio', 'whole': 'whole_image_radio'}
FEATURE_METHODS = {'blob': 'Blob Detection', 'corner': 'Corner Detection'}


def _check(dialog, attr):
    if attr and hasattr(dialog, attr):
        getattr(dialog, attr).setChecked(True)


def preset_projection(dialog, method=None):
    """Select the projection method ('max', 'maximum', ..., 'standard deviation')."""
    if method:
        _check(dialog, PROJECTION_RADIOS.get(str(method).lower()))


def preset_registration(dialog, method=None, reference_frame=None):
    """Select the registration method ('phase' or 'feature') and the reference frame."""
    if method in ('phase', 'feature'):
        _check(dialog, f'{method}_radio')
    if reference_frame is not None:
        dialog.ref_frame_spin.setValue(int(reference_frame))


def preset_analysis(dialog, roi_type=None, method=None, threshold=None, min_size=None, max_size=None):
    """Select the ROI source ('current', 'mask', 'whole') and, for feature detection, the method and sizes."""
    if roi_type:
        _check(dialog, ROI_RADIOS.get(roi_type))
    if method in FEATURE_METHODS and hasattr(dialog, 'method_combo'):
        dialog.method_combo.setCurrentText(FEATURE_METHODS[method])
    for value, attr in ((threshold, 'threshold_spin'), (min_size, 'min_size_spin'), (max_size, 'max_size_spin')):
        if value is not None and hasattr(dialog, attr):
            getattr(dialog, attr).setValue(float(value))

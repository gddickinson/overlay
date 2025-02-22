"""
Export and save functionality for the Advanced TIFF Stack Viewer.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
import cv2

class ExportManager:
    """Class for managing export and save operations."""

    def __init__(self, logger=None):
        """Initialize export manager."""
        self.logger = logger or logging.getLogger('tiff_stack_viewer')

    def save_image(self, image, file_path):
        """Save a single image to file."""
        self.logger.info(f"Saving image to {file_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Determine file type from extension
            ext = os.path.splitext(file_path)[1].lower()

            # Ensure image is in the right format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image, all good
                pass
            elif len(image.shape) == 2:
                # Grayscale image, convert to RGB if saving as JPEG
                if ext in ['.jpg', '.jpeg']:
                    image = np.stack([image] * 3, axis=2)

            # Save image
            io.imsave(file_path, image)

            return True

        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            return False

    def export_frames(self, fluorescence_stack, mask_stack, output_dir, format='png',
                      frames=None, apply_overlay=True, display_settings=None):
        """Export frames from stacks to image files."""
        self.logger.info(f"Exporting frames to {output_dir}")

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Determine frames to export
            if frames is None:
                # Export all frames
                max_frames = 0
                if fluorescence_stack:
                    max_frames = max(max_frames, fluorescence_stack.max_frames)
                if mask_stack:
                    max_frames = max(max_frames, mask_stack.max_frames)

                frames = list(range(max_frames))
            elif isinstance(frames, str):
                # Parse frame range
                if frames.lower() == 'all':
                    max_frames = 0
                    if fluorescence_stack:
                        max_frames = max(max_frames, fluorescence_stack.max_frames)
                    if mask_stack:
                        max_frames = max(max_frames, mask_stack.max_frames)

                    frames = list(range(max_frames))
                elif '-' in frames:
                    # Range like "1-10"
                    start, end = frames.split('-')
                    frames = list(range(int(start), int(end) + 1))
                elif ',' in frames:
                    # List like "1,3,5"
                    frames = [int(f) for f in frames.split(',')]
                else:
                    # Single frame
                    frames = [int(frames)]

            # Setup display settings
            if display_settings is None:
                display_settings = {
                    'fluorescence_visible': True,
                    'mask_visible': True,
                    'overlay_alpha': 0.5,
                    'mask_color': [255, 0, 0]
                }

            # Export each frame
            num_exported = 0
            for frame_idx in frames:
                # Get fluorescence frame
                fluor_frame = None
                if (fluorescence_stack and frame_idx < fluorescence_stack.max_frames and
                    display_settings['fluorescence_visible']):
                    fluor_frame = fluorescence_stack.get_normalized_frame(frame_idx)

                # Get mask frame
                mask_frame = None
                if (mask_stack and frame_idx < mask_stack.max_frames and
                    display_settings['mask_visible']):
                    mask_frame = mask_stack.get_frame(frame_idx)

                # Skip if no data for this frame
                if fluor_frame is None and mask_frame is None:
                    continue

                # Combine images if needed
                if apply_overlay and fluor_frame is not None and mask_frame is not None:
                    # Create RGB image from grayscale
                    if len(fluor_frame.shape) == 2:
                        display_image = np.stack([fluor_frame] * 3, axis=2)
                    else:
                        display_image = fluor_frame.copy()

                    # Apply overlay with color
                    binary_mask = mask_frame > 0

                    for i in range(3):
                        mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                        mask_channel[binary_mask] = display_settings['mask_color'][i]

                        # Alpha blend
                        display_image[..., i] = (
                            display_image[..., i] * (1 - display_settings['overlay_alpha'] * binary_mask) +
                            mask_channel * display_settings['overlay_alpha']
                        ).astype(np.uint8)
                elif fluor_frame is not None:
                    display_image = fluor_frame
                elif mask_frame is not None:
                    # Convert binary mask to RGB
                    mask_rgb = np.zeros((*mask_frame.shape, 3), dtype=np.uint8)
                    binary_mask = mask_frame > 0

                    for i in range(3):
                        mask_rgb[..., i][binary_mask] = display_settings['mask_color'][i]

                    display_image = mask_rgb
                else:
                    continue

                # Generate output filename
                output_file = os.path.join(output_dir, f"frame_{frame_idx:04d}.{format}")

                # Save image
                self.save_image(display_image, output_file)
                num_exported += 1

            self.logger.info(f"Exported {num_exported} frames to {output_dir}")
            return num_exported > 0

        except Exception as e:
            self.logger.error(f"Error exporting frames: {e}")
            return False

    def export_movie(self, fluorescence_stack, mask_stack, output_file, fps=10,
                     apply_overlay=True, display_settings=None):
        """Export stacks as a movie file."""
        self.logger.info(f"Exporting movie to {output_file}")

        try:
            # Determine max frames
            max_frames = 0
            if fluorescence_stack:
                max_frames = max(max_frames, fluorescence_stack.max_frames)
            if mask_stack:
                max_frames = max(max_frames, mask_stack.max_frames)

            if max_frames == 0:
                self.logger.error("No frames to export")
                return False

            # Setup display settings
            if display_settings is None:
                display_settings = {
                    'fluorescence_visible': True,
                    'mask_visible': True,
                    'overlay_alpha': 0.5,
                    'mask_color': [255, 0, 0]
                }

            # Get dimensions from first frame
            height, width = 0, 0
            if fluorescence_stack:
                first_frame = fluorescence_stack.get_normalized_frame(0)
                height, width = first_frame.shape[:2]
            elif mask_stack:
                first_frame = mask_stack.get_frame(0)
                height, width = first_frame.shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for mp4
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            # Process each frame
            for frame_idx in range(max_frames):
                # Get fluorescence frame
                fluor_frame = None
                if (fluorescence_stack and frame_idx < fluorescence_stack.max_frames and
                    display_settings['fluorescence_visible']):
                    fluor_frame = fluorescence_stack.get_normalized_frame(frame_idx)

                # Get mask frame
                mask_frame = None
                if (mask_stack and frame_idx < mask_stack.max_frames and
                    display_settings['mask_visible']):
                    mask_frame = mask_stack.get_frame(frame_idx)

                # Skip if no data for this frame
                if fluor_frame is None and mask_frame is None:
                    continue

                # Combine images if needed
                if apply_overlay and fluor_frame is not None and mask_frame is not None:
                    # Create RGB image from grayscale
                    if len(fluor_frame.shape) == 2:
                        display_image = np.stack([fluor_frame] * 3, axis=2)
                    else:
                        display_image = fluor_frame.copy()

                    # Apply overlay with color
                    binary_mask = mask_frame > 0

                    for i in range(3):
                        mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                        mask_channel[binary_mask] = display_settings['mask_color'][i]

                        # Alpha blend
                        display_image[..., i] = (
                            display_image[..., i] * (1 - display_settings['overlay_alpha'] * binary_mask) +
                            mask_channel * display_settings['overlay_alpha']
                        ).astype(np.uint8)
                elif fluor_frame is not None:
                    if len(fluor_frame.shape) == 2:
                        display_image = np.stack([fluor_frame] * 3, axis=2)
                    else:
                        display_image = fluor_frame
                elif mask_frame is not None:
                    # Convert binary mask to RGB
                    mask_rgb = np.zeros((*mask_frame.shape, 3), dtype=np.uint8)
                    binary_mask = mask_frame > 0

                    for i in range(3):
                        mask_rgb[..., i][binary_mask] = display_settings['mask_color'][i]

                    display_image = mask_rgb
                else:
                    continue

                # Convert to BGR for OpenCV
                if len(display_image.shape) == 3 and display_image.shape[2] == 3:
                    display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                else:
                    display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

                # Write frame
                out.write(display_image_bgr)

            # Release video writer
            out.release()

            self.logger.info(f"Exported movie with {max_frames} frames to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting movie: {e}")
            return False

    def export_data(self, fluorescence_stack, mask_stack, output_file, include_stats=True,
                    roi_data=True, rois=None):
        """Export numerical data to CSV file."""
        self.logger.info(f"Exporting data to {output_file}")

        try:
            # Determine max frames
            max_frames = 0
            if fluorescence_stack:
                max_frames = max(max_frames, fluorescence_stack.max_frames)
            if mask_stack:
                max_frames = max(max_frames, mask_stack.max_frames)

            if max_frames == 0:
                self.logger.error("No frames to export")
                return False

            # Initialize data dictionary
            data = {
                'Frame': list(range(max_frames))
            }

            # Add basic statistics if requested
            if include_stats and fluorescence_stack:
                for i in range(max_frames):
                    if i < fluorescence_stack.max_frames:
                        frame = fluorescence_stack.get_frame(i)

                        if i == 0:
                            # Initialize lists for stats
                            data['Mean'] = []
                            data['Median'] = []
                            data['StdDev'] = []
                            data['Min'] = []
                            data['Max'] = []

                        # Calculate and store stats
                        data['Mean'].append(np.mean(frame))
                        data['Median'].append(np.median(frame))
                        data['StdDev'].append(np.std(frame))
                        data['Min'].append(np.min(frame))
                        data['Max'].append(np.max(frame))
                    else:
                        # Add NaN for missing frames
                        if i == 0:
                            # Initialize lists for stats
                            data['Mean'] = []
                            data['Median'] = []
                            data['StdDev'] = []
                            data['Min'] = []
                            data['Max'] = []

                        data['Mean'].append(np.nan)
                        data['Median'].append(np.nan)
                        data['StdDev'].append(np.nan)
                        data['Min'].append(np.nan)
                        data['Max'].append(np.nan)

            # Add ROI data if requested
            if roi_data and fluorescence_stack and rois:
                for roi_id, roi_info in rois.items():
                    # Extract ROI stats for each frame
                    roi_mean = []
                    roi_sum = []

                    for i in range(max_frames):
                        if i < fluorescence_stack.max_frames:
                            frame = fluorescence_stack.get_frame(i)

                            # Create mask from ROI
                            roi_mask = np.zeros(frame.shape, dtype=bool)

                            if roi_info['type'] == 'rectangle':
                                # Rectangle [y1, x1, y2, x2]
                                y1, x1, y2, x2 = roi_info['points']
                                roi_mask[y1:y2, x1:x2] = True

                            elif roi_info['type'] == 'ellipse':
                                # Ellipse [center_y, center_x, radius_y, radius_x]
                                cy, cx, ry, rx = roi_info['points']
                                y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
                                dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                                roi_mask[dist <= 1] = True

                            elif roi_info['type'] == 'polygon':
                                # Convert polygon to mask using OpenCV
                                points = np.array(roi_info['points'], dtype=np.int32)
                                mask_img = np.zeros(frame.shape, dtype=np.uint8)
                                cv2.fillPoly(mask_img, [points], 1)
                                roi_mask = mask_img.astype(bool)

                            # Calculate stats
                            roi_pixels = frame[roi_mask]
                            if len(roi_pixels) > 0:
                                roi_mean.append(np.mean(roi_pixels))
                                roi_sum.append(np.sum(roi_pixels))
                            else:
                                roi_mean.append(np.nan)
                                roi_sum.append(np.nan)
                        else:
                            roi_mean.append(np.nan)
                            roi_sum.append(np.nan)

                    # Add to data dictionary
                    data[f'{roi_id}_Mean'] = roi_mean
                    data[f'{roi_id}_Sum'] = roi_sum

            # Create DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)

            self.logger.info(f"Exported data to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False

    def export_analysis_results(self, results, output_dir, prefix=None, include_plots=True):
        """Export analysis results to files."""
        self.logger.info(f"Exporting analysis results to {output_dir}")

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Set prefix if not provided
            if prefix is None:
                prefix = f"analysis_{timestamp}"

            # Export data to CSV
            csv_file = os.path.join(output_dir, f"{prefix}_data.csv")

            # Determine type of results and format accordingly
            if 'metrics' in results and 'time_points' in results:
                # Time series analysis results
                data = {'Frame': results['time_points']}

                for roi_id, metrics in results['metrics'].items():
                    for metric_name, values in metrics.items():
                        data[f"{roi_id}_{metric_name}"] = values

                # Create DataFrame and save
                df = pd.DataFrame(data)
                df.to_csv(csv_file, index=False)

                # Create plot if requested
                if include_plots and 'mean' in results:
                    # Create time series plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    for roi_id, metrics in results['metrics'].items():
                        if 'mean' in metrics:
                            ax.plot(results['time_points'], metrics['mean'], label=roi_id)

                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Mean Intensity')
                    ax.set_title('Time Series Analysis')
                    ax.legend()
                    ax.grid(True)

                    # Save plot
                    plot_file = os.path.join(output_dir, f"{prefix}_plot.png")
                    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)

            else:
                # Other results
                # Try to convert results to DataFrame
                try:
                    df = pd.DataFrame(results)
                    df.to_csv(csv_file, index=False)
                except Exception:
                    # If conversion fails, save as JSON
                    import json
                    json_file = os.path.join(output_dir, f"{prefix}_data.json")
                    with open(json_file, 'w') as f:
                        json.dump(results, f, indent=2)

            self.logger.info(f"Exported analysis results to {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting analysis results: {e}")
            return False

    def create_report(self, fluorescence_stack, mask_stack, analysis_results, rois, output_file):
        """Create a comprehensive HTML report."""
        self.logger.info(f"Creating report at {output_file}")

        try:
            # Create directories
            report_dir = os.path.dirname(output_file)
            images_dir = os.path.join(report_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Start HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>TIFF Stack Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .section {{ margin-bottom: 30px; }}
                    .image-container {{ margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>TIFF Stack Analysis Report</h1>
                <p>Generated on {timestamp}</p>

                <div class="section">
                    <h2>Data Summary</h2>
            """

            # Add data summary
            if fluorescence_stack:
                html_content += f"""
                    <h3>Fluorescence Stack</h3>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>Filename</td><td>{fluorescence_stack.metadata.get('filename', 'N/A')}</td></tr>
                        <tr><td>Frames</td><td>{fluorescence_stack.max_frames}</td></tr>
                        <tr><td>Dimensions</td><td>{fluorescence_stack.dimensions['height']} × {fluorescence_stack.dimensions['width']}</td></tr>
                        <tr><td>Data Type</td><td>{fluorescence_stack.metadata.get('dtype', 'N/A')}</td></tr>
                    </table>
                """

            if mask_stack:
                html_content += f"""
                    <h3>Mask Stack</h3>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>Filename</td><td>{mask_stack.metadata.get('filename', 'N/A')}</td></tr>
                        <tr><td>Frames</td><td>{mask_stack.max_frames}</td></tr>
                        <tr><td>Dimensions</td><td>{mask_stack.dimensions['height']} × {mask_stack.dimensions['width']}</td></tr>
                        <tr><td>Data Type</td><td>{mask_stack.metadata.get('dtype', 'N/A')}</td></tr>
                    </table>
                """

            # Add ROI information
            if rois:
                html_content += """
                    <h3>Regions of Interest</h3>
                    <table>
                        <tr><th>ROI ID</th><th>Type</th><th>Coordinates</th></tr>
                """

                for roi_id, roi_info in rois.items():
                    html_content += f"""
                        <tr>
                            <td>{roi_id}</td>
                            <td>{roi_info['type']}</td>
                            <td>{roi_info['points']}</td>
                        </tr>
                    """

                html_content += """
                    </table>
                """

            # Add analysis results
            if analysis_results:
                html_content += """
                    <div class="section">
                        <h2>Analysis Results</h2>
                """

                # Determine type of results and format accordingly
                if 'metrics' in analysis_results and 'time_points' in analysis_results:
                    # Time series analysis
                    html_content += """
                        <h3>Time Series Analysis</h3>
                        <div class="image-container">
                    """

                    # Create and save time series plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    for roi_id, metrics in analysis_results['metrics'].items():
                        if 'mean' in metrics:
                            ax.plot(analysis_results['time_points'], metrics['mean'], label=roi_id)

                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Mean Intensity')
                    ax.set_title('Time Series Analysis')
                    ax.legend()
                    ax.grid(True)

                    # Save plot
                    plot_file = os.path.join(images_dir, 'time_series_plot.png')
                    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    html_content += f"""
                            <img src="images/time_series_plot.png" alt="Time Series Plot">
                        </div>

                        <h4>Trends and Peaks</h4>
                    """

                    # Add trend and peak information if available
                    if 'trend' in analysis_results:
                        trend = analysis_results['trend']
                        html_content += f"""
                            <p>Trend: {trend['trend']}</p>
                            <p>Slope: {trend['slope']:.4f}</p>
                            <p>R-value: {trend['r_value']:.4f}</p>
                            <p>P-value: {trend['p_value']:.4f}</p>
                        """

                    if 'peaks' in analysis_results:
                        peaks = analysis_results['peaks']
                        html_content += f"""
                            <p>Number of peaks: {peaks['num_peaks']}</p>
                            <p>Peak locations: {', '.join(map(str, peaks['peak_indices']))}</p>
                        """

                elif 'correlation_matrix' in analysis_results:
                    # Correlation analysis
                    html_content += """
                        <h3>Correlation Analysis</h3>
                        <div class="image-container">
                    """

                    # Create and save correlation matrix plot
                    corr_matrix = np.array(analysis_results['correlation_matrix'])
                    roi_ids = analysis_results['roi_ids']

                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    fig.colorbar(im, ax=ax)

                    ax.set_xticks(np.arange(len(roi_ids)))
                    ax.set_yticks(np.arange(len(roi_ids)))
                    ax.set_xticklabels(roi_ids, rotation=45, ha='right')
                    ax.set_yticklabels(roi_ids)

                    ax.set_title('ROI Correlation Matrix')
                    fig.tight_layout()

                    # Save plot
                    plot_file = os.path.join(images_dir, 'correlation_matrix.png')
                    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    html_content += f"""
                            <img src="images/correlation_matrix.png" alt="Correlation Matrix">
                        </div>
                    """

                elif 'components' in analysis_results:
                    # PCA results
                    html_content += """
                        <h3>Principal Component Analysis</h3>
                        <div class="image-container">
                    """

                    # Create and save variance plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(1, len(analysis_results['explained_variance']) + 1),
                           analysis_results['explained_variance'])
                    ax.set_xlabel('Principal Component')
                    ax.set_ylabel('Explained Variance Ratio')
                    ax.set_title('PCA Explained Variance')
                    ax.set_xticks(range(1, len(analysis_results['explained_variance']) + 1))

                    # Save plot
                    plot_file = os.path.join(images_dir, 'pca_variance.png')
                    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    html_content += f"""
                            <img src="images/pca_variance.png" alt="PCA Explained Variance">
                        </div>
                    """

                html_content += """
                    </div>
                """

            # Add sample images
            html_content += """
                <div class="section">
                    <h2>Sample Images</h2>
            """

            # Save sample frames (first, middle, last)
            if fluorescence_stack:
                frames_to_save = [0, fluorescence_stack.max_frames // 2, fluorescence_stack.max_frames - 1]

                for i, frame_idx in enumerate(frames_to_save):
                    if frame_idx < fluorescence_stack.max_frames:
                        frame = fluorescence_stack.get_normalized_frame(frame_idx)

                        # Apply mask overlay if available
                        if mask_stack and frame_idx < mask_stack.max_frames:
                            mask = mask_stack.get_frame(frame_idx)

                            # Create RGB image from grayscale
                            if len(frame.shape) == 2:
                                display_image = np.stack([frame] * 3, axis=2)
                            else:
                                display_image = frame.copy()

                            # Apply overlay with color
                            binary_mask = mask > 0
                            overlay_alpha = 0.5
                            mask_color = [255, 0, 0]  # Default red

                            for i in range(3):
                                mask_channel = np.zeros_like(binary_mask, dtype=np.uint8)
                                mask_channel[binary_mask] = mask_color[i]

                                # Alpha blend
                                display_image[..., i] = (
                                    display_image[..., i] * (1 - overlay_alpha * binary_mask) +
                                    mask_channel * overlay_alpha
                                ).astype(np.uint8)
                        else:
                            display_image = frame

                        # Save image
                        img_file = os.path.join(images_dir, f"frame_{frame_idx:04d}.png")
                        io.imsave(img_file, display_image)

                        html_content += f"""
                            <div class="image-container">
                                <h3>Frame {frame_idx+1}</h3>
                                <img src="images/frame_{frame_idx:04d}.png" alt="Frame {frame_idx+1}">
                            </div>
                        """

            # Close HTML
            html_content += """
                </div>
            </body>
            </html>
            """

            # Write HTML to file
            with open(output_file, 'w') as f:
                f.write(html_content)

            self.logger.info(f"Created report at {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating report: {e}")
            return False

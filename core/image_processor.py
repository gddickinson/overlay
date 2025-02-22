"""
Image processing functionality for TIFF stacks.
"""

import logging
import numpy as np
from scipy import ndimage
from skimage import filters, exposure, registration, segmentation, morphology, feature
import cv2


class ImageProcessor:
    """Class for processing image stacks."""
    
    def __init__(self, logger=None):
        """Initialize image processor."""
        self.logger = logger or logging.getLogger('tiff_stack_viewer')
    
    def apply_filter(self, image, filter_type, **kwargs):
        """Apply various filters to an image."""
        self.logger.info(f"Applying {filter_type} filter")
        
        try:
            if filter_type == 'gaussian':
                sigma = kwargs.get('sigma', 1.0)
                return filters.gaussian(image, sigma=sigma, preserve_range=True).astype(image.dtype)
            
            elif filter_type == 'median':
                size = kwargs.get('size', 3)
                return filters.median(image, selem=morphology.disk(size), preserve_range=True).astype(image.dtype)
            
            elif filter_type == 'bilateral':
                sigma_color = kwargs.get('sigma_color', 0.1)
                sigma_spatial = kwargs.get('sigma_spatial', 1)
                return filters.gaussian(image, sigma=sigma_spatial, preserve_range=True).astype(image.dtype)
            
            elif filter_type == 'sobel':
                return filters.sobel(image)
            
            elif filter_type == 'laplacian':
                return filters.laplace(image)
            
            elif filter_type == 'frangi':
                scale_range = kwargs.get('scale_range', (1, 10))
                scale_step = kwargs.get('scale_step', 2)
                return filters.frangi(image, scale_range=scale_range, scale_step=scale_step)
            
            else:
                self.logger.error(f"Unknown filter type: {filter_type}")
                return image
                
        except Exception as e:
            self.logger.error(f"Error applying {filter_type} filter: {e}")
            return image
    
    def enhance_contrast(self, image, method='rescale', **kwargs):
        """Enhance image contrast using various methods."""
        self.logger.info(f"Enhancing contrast using {method}")
        
        try:
            if method == 'rescale':
                p_low = kwargs.get('p_low', 2)
                p_high = kwargs.get('p_high', 98)
                
                # Calculate percentiles
                low = np.percentile(image, p_low)
                high = np.percentile(image, p_high)
                
                # Apply contrast stretching
                return exposure.rescale_intensity(
                    image, in_range=(low, high), out_range=(image.min(), image.max())
                )
            
            elif method == 'equalize':
                return exposure.equalize_hist(image)
            
            elif method == 'adaptive':
                kernel_size = kwargs.get('kernel_size', 64)
                clip_limit = kwargs.get('clip_limit', 0.01)
                return exposure.equalize_adapthist(
                    image, kernel_size=kernel_size, clip_limit=clip_limit
                )
            
            else:
                self.logger.error(f"Unknown contrast enhancement method: {method}")
                return image
                
        except Exception as e:
            self.logger.error(f"Error enhancing contrast with {method}: {e}")
            return image
    
    def register_images(self, fixed_image, moving_image, method='phase', **kwargs):
        """Register a moving image to a fixed reference image."""
        self.logger.info(f"Registering images using {method}")
        
        try:
            if method == 'phase':
                # Phase correlation
                shift, error, diffphase = registration.phase_cross_correlation(
                    fixed_image, moving_image, upsample_factor=kwargs.get('upsample_factor', 10)
                )
                
                # Apply transformation
                registered = ndimage.shift(moving_image, shift, mode='constant', cval=0)
                return registered, shift
            
            elif method == 'feature':
                # Feature-based registration using ORB
                # Convert to 8-bit if needed
                fixed_8bit = self._convert_to_8bit(fixed_image)
                moving_8bit = self._convert_to_8bit(moving_image)
                
                # Use OpenCV ORB for feature detection
                orb = cv2.ORB_create()
                
                # Detect keypoints and compute descriptors
                kp1, des1 = orb.detectAndCompute(fixed_8bit, None)
                kp2, des2 = orb.detectAndCompute(moving_8bit, None)
                
                if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                    self.logger.warning("Not enough features detected for registration")
                    return moving_image, None
                
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Take top matches
                good_matches = matches[:min(50, len(matches))]
                
                # Extract locations of matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    self.logger.warning("Could not compute homography for registration")
                    return moving_image, None
                
                # Apply transformation
                height, width = fixed_image.shape
                registered = cv2.warpPerspective(moving_image, H, (width, height))
                
                return registered, H
            
            else:
                self.logger.error(f"Unknown registration method: {method}")
                return moving_image, None
                
        except Exception as e:
            self.logger.error(f"Error in image registration: {e}")
            return moving_image, None
    
    def segment_image(self, image, method='threshold', **kwargs):
        """Segment an image using various methods."""
        self.logger.info(f"Segmenting image using {method}")
        
        try:
            if method == 'threshold':
                threshold = kwargs.get('threshold', None)
                if threshold is None:
                    # Automatic threshold
                    threshold = filters.threshold_otsu(image)
                
                return image > threshold
            
            elif method == 'adaptive':
                block_size = kwargs.get('block_size', 35)
                offset = kwargs.get('offset', 0)
                
                return filters.threshold_local(
                    image, block_size=block_size, offset=offset
                )
            
            elif method == 'watershed':
                # Calculate gradient
                gradient = filters.sobel(image)
                
                # Marker creation
                low = kwargs.get('low_marker', 30)
                high = kwargs.get('high_marker', 150)
                
                markers = np.zeros_like(image, dtype=int)
                markers[image < low] = 1
                markers[image > high] = 2
                
                # Apply watershed
                segmented = segmentation.watershed(gradient, markers)
                return segmented - 1  # Convert to binary (0 and 1)
            
            else:
                self.logger.error(f"Unknown segmentation method: {method}")
                return image > filters.threshold_otsu(image)
                
        except Exception as e:
            self.logger.error(f"Error in image segmentation: {e}")
            return image > filters.threshold_otsu(image)
    
    def detect_features(self, image, method='blob', **kwargs):
        """Detect features in an image."""
        self.logger.info(f"Detecting features using {method}")
        
        try:
            if method == 'blob':
                # Blob detection
                min_sigma = kwargs.get('min_sigma', 1)
                max_sigma = kwargs.get('max_sigma', 10)
                num_sigma = kwargs.get('num_sigma', 5)
                threshold = kwargs.get('threshold', 0.1)
                
                blobs = feature.blob_log(
                    image, 
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold
                )
                
                return blobs
            
            elif method == 'corner':
                # Corner detection
                min_distance = kwargs.get('min_distance', 5)
                threshold = kwargs.get('threshold', 0.1)
                
                corners = feature.corner_peaks(
                    feature.corner_harris(image),
                    min_distance=min_distance,
                    threshold_rel=threshold
                )
                
                return corners
            
            else:
                self.logger.error(f"Unknown feature detection method: {method}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error in feature detection: {e}")
            return []
    
    def overlay_images(self, base_image, overlay_image, alpha=0.5, color=None):
        """Overlay one image on top of another with opacity."""
        self.logger.info(f"Overlaying images with alpha={alpha}")
        
        try:
            # Convert to float for calculation
            base_float = base_image.astype(float)
            overlay_float = overlay_image.astype(float)
            
            # Apply color to overlay if specified
            if color is not None:
                # Create RGB overlay
                rgb_overlay = np.zeros((*overlay_image.shape, 3), dtype=float)
                for i, c in enumerate(color[:3]):  # Use first 3 values as RGB
                    rgb_overlay[..., i] = overlay_float * (c / 255.0)
                
                # Create RGB base if it's grayscale
                if len(base_image.shape) == 2:
                    rgb_base = np.stack([base_float] * 3, axis=-1)
                else:
                    rgb_base = base_float
                
                # Blend
                blended = rgb_base * (1 - alpha * overlay_float) + rgb_overlay * (alpha * overlay_float)
                return np.clip(blended, 0, 255).astype(np.uint8)
                
            else:
                # Simple alpha blending
                blended = base_float * (1 - alpha) + overlay_float * alpha
                return np.clip(blended, 0, 255).astype(np.uint8)
                
        except Exception as e:
            self.logger.error(f"Error overlaying images: {e}")
            return base_image
    
    def measure_intensity(self, image, mask=None, regions=None):
        """Measure intensity statistics in the image, optionally within a mask or regions."""
        try:
            if mask is not None:
                # Measure within mask
                masked_data = image[mask > 0]
                if len(masked_data) == 0:
                    return None
                
                stats = {
                    'mean': np.mean(masked_data),
                    'std': np.std(masked_data),
                    'min': np.min(masked_data),
                    'max': np.max(masked_data),
                    'sum': np.sum(masked_data),
                    'count': len(masked_data)
                }
                return stats
                
            elif regions is not None:
                # Measure within specified regions
                results = []
                for region in regions:
                    if 'slice' in region:
                        # Use predefined slice
                        region_data = image[region['slice']]
                    elif 'points' in region:
                        # Create mask from points
                        r_mask = np.zeros_like(image, dtype=bool)
                        points = region['points']
                        if region['type'] == 'polygon':
                            # Fill polygon
                            r_mask = self._fill_polygon(r_mask, points)
                        elif region['type'] == 'rectangle':
                            # Fill rectangle (y1, x1, y2, x2)
                            y1, x1, y2, x2 = points
                            r_mask[y1:y2, x1:x2] = True
                        elif region['type'] == 'circle':
                            # Fill circle (y, x, radius)
                            y, x, r = points
                            Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                            dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
                            r_mask[dist <= r] = True
                        
                        region_data = image[r_mask]
                    else:
                        continue
                    
                    if len(region_data) == 0:
                        results.append(None)
                        continue
                    
                    # Calculate statistics
                    stats = {
                        'mean': np.mean(region_data),
                        'std': np.std(region_data),
                        'min': np.min(region_data),
                        'max': np.max(region_data),
                        'sum': np.sum(region_data),
                        'count': len(region_data)
                    }
                    
                    # Add region identifier
                    if 'id' in region:
                        stats['id'] = region['id']
                    
                    results.append(stats)
                
                return results
                
            else:
                # Measure over entire image
                stats = {
                    'mean': np.mean(image),
                    'std': np.std(image),
                    'min': np.min(image),
                    'max': np.max(image),
                    'sum': np.sum(image),
                    'count': image.size
                }
                return stats
                
        except Exception as e:
            self.logger.error(f"Error measuring intensity: {e}")
            return None
    
    def extract_time_series(self, image_stack, mask=None, regions=None):
        """Extract time series data from an image stack."""
        self.logger.info("Extracting time series data")
        
        try:
            if image_stack is None or image_stack.data is None:
                return None
            
            num_frames = image_stack.dimensions['frames']
            time_points = list(range(num_frames))
            
            # Prepare results container
            if regions is not None:
                num_regions = len(regions)
                series_data = {
                    'time': time_points,
                    'mean': [[] for _ in range(num_regions)],
                    'std': [[] for _ in range(num_regions)],
                    'sum': [[] for _ in range(num_regions)],
                    'max': [[] for _ in range(num_regions)],
                    'min': [[] for _ in range(num_regions)]
                }
                
                # Add region identifiers
                region_ids = []
                for region in regions:
                    if 'id' in region:
                        region_ids.append(region['id'])
                    else:
                        region_ids.append(f"Region {len(region_ids)+1}")
                series_data['region_ids'] = region_ids
                
            elif mask is not None:
                series_data = {
                    'time': time_points,
                    'mean': [],
                    'std': [],
                    'sum': [],
                    'max': [],
                    'min': []
                }
            else:
                series_data = {
                    'time': time_points,
                    'mean': [],
                    'std': [],
                    'sum': [],
                    'max': [],
                    'min': []
                }
            
            # Process each frame
            for i in range(num_frames):
                frame = image_stack.get_frame(i)
                
                if regions is not None:
                    stats_list = self.measure_intensity(frame, regions=regions)
                    for j, stats in enumerate(stats_list):
                        if stats:
                            series_data['mean'][j].append(stats['mean'])
                            series_data['std'][j].append(stats['std'])
                            series_data['sum'][j].append(stats['sum'])
                            series_data['max'][j].append(stats['max'])
                            series_data['min'][j].append(stats['min'])
                        else:
                            series_data['mean'][j].append(None)
                            series_data['std'][j].append(None)
                            series_data['sum'][j].append(None)
                            series_data['max'][j].append(None)
                            series_data['min'][j].append(None)
                
                elif mask is not None:
                    stats = self.measure_intensity(frame, mask=mask)
                    if stats:
                        series_data['mean'].append(stats['mean'])
                        series_data['std'].append(stats['std'])
                        series_data['sum'].append(stats['sum'])
                        series_data['max'].append(stats['max'])
                        series_data['min'].append(stats['min'])
                    else:
                        series_data['mean'].append(None)
                        series_data['std'].append(None)
                        series_data['sum'].append(None)
                        series_data['max'].append(None)
                        series_data['min'].append(None)
                
                else:
                    stats = self.measure_intensity(frame)
                    series_data['mean'].append(stats['mean'])
                    series_data['std'].append(stats['std'])
                    series_data['sum'].append(stats['sum'])
                    series_data['max'].append(stats['max'])
                    series_data['min'].append(stats['min'])
            
            return series_data
            
        except Exception as e:
            self.logger.error(f"Error extracting time series: {e}")
            return None
    
    def _convert_to_8bit(self, image):
        """Convert image to 8-bit for compatibility with OpenCV functions."""
        if image.dtype == np.uint8:
            return image
        
        # Normalize to 0-255
        img_min = image.min()
        img_max = image.max()
        
        # Avoid division by zero
        if img_max == img_min:
            img_max = img_min + 1
        
        img_8bit = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return img_8bit
    
    def _fill_polygon(self, mask, points):
        """Fill a polygon in a mask given by points."""
        # Convert points to format expected by cv2
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Create an empty image to draw the polygon
        img = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(img, [pts], 1)
        
        # Update the mask
        mask[img > 0] = True
        return mask

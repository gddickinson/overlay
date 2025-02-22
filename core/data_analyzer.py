"""
Data analysis functionality for TIFF stacks.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn import decomposition, cluster


class DataAnalyzer:
    """Class for analyzing image data and time series."""
    
    def __init__(self, logger=None):
        """Initialize data analyzer."""
        self.logger = logger or logging.getLogger('tiff_stack_viewer')
    
    def calculate_basic_stats(self, image_stack):
        """Calculate basic statistics for each frame in a stack."""
        self.logger.info("Calculating basic statistics")
        
        try:
            if image_stack is None or image_stack.data is None:
                self.logger.error("No data available for statistics calculation")
                return None
            
            # Get data and dimensions
            data = image_stack.data
            num_frames = image_stack.dimensions['frames']
            
            # Initialize results
            stats = {
                'frame': list(range(num_frames)),
                'mean': [],
                'median': [],
                'std': [],
                'min': [],
                'max': [],
                'snr': []
            }
            
            # Calculate statistics for each frame
            for i in range(num_frames):
                frame = data[i]
                
                stats['mean'].append(np.mean(frame))
                stats['median'].append(np.median(frame))
                stats['std'].append(np.std(frame))
                stats['min'].append(np.min(frame))
                stats['max'].append(np.max(frame))
                
                # Calculate signal-to-noise ratio (SNR)
                if np.std(frame) > 0:
                    snr = np.mean(frame) / np.std(frame)
                    stats['snr'].append(snr)
                else:
                    stats['snr'].append(0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return None
    
    def analyze_roi_time_series(self, time_series, roi_id=None):
        """Analyze a time series from an ROI."""
        self.logger.info(f"Analyzing time series for ROI: {roi_id or 'all'}")
        
        try:
            if not time_series:
                self.logger.error("No time series data available for analysis")
                return None
            
            # Create result container
            results = {
                'time_points': time_series['time'],
                'metrics': {}
            }
            
            # If analyzing a specific ROI
            if roi_id is not None:
                if 'region_ids' in time_series:
                    # Find the index of the ROI
                    try:
                        roi_index = time_series['region_ids'].index(roi_id)
                    except ValueError:
                        self.logger.error(f"ROI {roi_id} not found in time series data")
                        return None
                    
                    # Extract data for this ROI
                    metrics = {}
                    for metric in ['mean', 'std', 'sum', 'max', 'min']:
                        if metric in time_series:
                            metrics[metric] = time_series[metric][roi_index]
                    
                    results['metrics'][roi_id] = metrics
                    
                    # Analyze trends
                    trend_results = self._analyze_trend(metrics['mean'])
                    results['trend'] = trend_results
                    
                    # Find peaks
                    peaks = self._find_peaks(metrics['mean'])
                    results['peaks'] = peaks
                    
                else:
                    # Single ROI data
                    metrics = {}
                    for metric in ['mean', 'std', 'sum', 'max', 'min']:
                        if metric in time_series:
                            metrics[metric] = time_series[metric]
                    
                    results['metrics'][roi_id or 'roi'] = metrics
                    
                    # Analyze trends
                    trend_results = self._analyze_trend(metrics['mean'])
                    results['trend'] = trend_results
                    
                    # Find peaks
                    peaks = self._find_peaks(metrics['mean'])
                    results['peaks'] = peaks
            
            else:
                # Analyze all ROIs
                if 'region_ids' in time_series:
                    for i, roi_id in enumerate(time_series['region_ids']):
                        metrics = {}
                        for metric in ['mean', 'std', 'sum', 'max', 'min']:
                            if metric in time_series:
                                metrics[metric] = time_series[metric][i]
                        
                        results['metrics'][roi_id] = metrics
                        
                        # Analyze trends for this ROI
                        trend_results = self._analyze_trend(metrics['mean'])
                        results.setdefault('trends', {})[roi_id] = trend_results
                        
                        # Find peaks for this ROI
                        peaks = self._find_peaks(metrics['mean'])
                        results.setdefault('peaks', {})[roi_id] = peaks
                
                else:
                    # Single ROI data
                    metrics = {}
                    for metric in ['mean', 'std', 'sum', 'max', 'min']:
                        if metric in time_series:
                            metrics[metric] = time_series[metric]
                    
                    results['metrics']['roi'] = metrics
                    
                    # Analyze trends
                    trend_results = self._analyze_trend(metrics['mean'])
                    results['trend'] = trend_results
                    
                    # Find peaks
                    peaks = self._find_peaks(metrics['mean'])
                    results['peaks'] = peaks
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing time series: {e}")
            return None
    
    def _analyze_trend(self, data):
        """Analyze trend in time series data."""
        # Convert to numpy array and remove NaN values
        data_array = np.array(data)
        valid_indices = ~np.isnan(data_array)
        data_clean = data_array[valid_indices]
        
        if len(data_clean) < 2:
            return {
                'slope': 0,
                'intercept': 0,
                'r_value': 0,
                'p_value': 1,
                'std_err': 0,
                'trend': 'insufficient data'
            }
        
        # Create time points array
        time_points = np.arange(len(data_clean))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, data_clean)
        
        # Determine trend
        if p_value <= 0.05:
            if slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no significant trend'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'trend': trend
        }
    
    def _find_peaks(self, data):
        """Find peaks in time series data."""
        # Convert to numpy array and remove NaN values
        data_array = np.array(data)
        valid_indices = ~np.isnan(data_array)
        data_clean = data_array[valid_indices]
        
        if len(data_clean) < 3:
            return {
                'peak_indices': [],
                'peak_values': [],
                'num_peaks': 0
            }
        
        # Find peaks
        peak_indices, _ = signal.find_peaks(data_clean, height=0, distance=2)
        
        # Get peak values
        peak_values = data_clean[peak_indices]
        
        # Map peak indices back to original indices
        original_indices = np.arange(len(data_array))[valid_indices][peak_indices]
        
        return {
            'peak_indices': original_indices.tolist(),
            'peak_values': peak_values.tolist(),
            'num_peaks': len(peak_indices)
        }
    
    def perform_pca(self, image_stack, n_components=3):
        """Perform Principal Component Analysis on image stack."""
        self.logger.info(f"Performing PCA with {n_components} components")
        
        try:
            if image_stack is None or image_stack.data is None:
                self.logger.error("No data available for PCA")
                return None
            
            # Get data and dimensions
            data = image_stack.data
            num_frames, height, width = data.shape[:3]
            
            # Reshape data for PCA: (frames, pixels)
            data_2d = data.reshape(num_frames, -1)
            
            # Create PCA object
            pca = decomposition.PCA(n_components=min(n_components, num_frames))
            
            # Fit and transform data
            components = pca.fit_transform(data_2d)
            
            # Get explained variance
            explained_variance = pca.explained_variance_ratio_
            
            # Reshape components back to images
            component_images = []
            for i in range(pca.n_components_):
                component = pca.components_[i].reshape(height, width)
                component_images.append(component)
            
            return {
                'components': components,
                'component_images': component_images,
                'explained_variance': explained_variance,
                'n_components': pca.n_components_
            }
            
        except Exception as e:
            self.logger.error(f"Error performing PCA: {e}")
            return None
    
    def cluster_time_series(self, time_series, n_clusters=3):
        """Cluster time series data."""
        self.logger.info(f"Clustering time series with {n_clusters} clusters")
        
        try:
            if not time_series or 'region_ids' not in time_series:
                self.logger.error("No suitable time series data available for clustering")
                return None
            
            # Extract mean values for all ROIs
            means = np.array(time_series['mean'])
            
            # Handle NaN values
            means[np.isnan(means)] = 0
            
            # Apply K-means clustering
            kmeans = cluster.KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(means)
            
            # Group ROIs by cluster
            clusters = {}
            for i, label in enumerate(labels):
                cluster_id = f"Cluster {label+1}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append(time_series['region_ids'][i])
            
            # Calculate cluster centroids (average time series for each cluster)
            centroids = {}
            for label, centroid in enumerate(kmeans.cluster_centers_):
                cluster_id = f"Cluster {label+1}"
                centroids[cluster_id] = centroid.tolist()
            
            return {
                'clusters': clusters,
                'centroids': centroids,
                'labels': labels.tolist(),
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            self.logger.error(f"Error clustering time series: {e}")
            return None
    
    def create_heatmap_data(self, time_series):
        """Create data for heatmap visualization of multiple time series."""
        self.logger.info("Creating heatmap data")
        
        try:
            if not time_series or 'region_ids' not in time_series:
                self.logger.error("No suitable time series data available for heatmap")
                return None
            
            # Extract data for heatmap
            roi_ids = time_series['region_ids']
            time_points = time_series['time']
            mean_values = np.array(time_series['mean'])
            
            # Normalize data for each ROI
            normalized_data = np.zeros_like(mean_values, dtype=float)
            for i in range(len(roi_ids)):
                row = mean_values[i]
                row_min = np.nanmin(row)
                row_max = np.nanmax(row)
                
                if row_max > row_min:
                    normalized_data[i] = (row - row_min) / (row_max - row_min)
                else:
                    normalized_data[i] = np.zeros_like(row)
            
            return {
                'roi_ids': roi_ids,
                'time_points': time_points,
                'values': mean_values.tolist(),
                'normalized': normalized_data.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap data: {e}")
            return None
    
    def calculate_correlation_matrix(self, time_series):
        """Calculate correlation matrix between ROI time series."""
        self.logger.info("Calculating correlation matrix")
        
        try:
            if not time_series or 'region_ids' not in time_series:
                self.logger.error("No suitable time series data available for correlation")
                return None
            
            # Extract data for correlation
            roi_ids = time_series['region_ids']
            mean_values = np.array(time_series['mean'])
            
            # Calculate correlation matrix
            corr_matrix = np.zeros((len(roi_ids), len(roi_ids)))
            
            for i in range(len(roi_ids)):
                for j in range(len(roi_ids)):
                    # Get time series for each ROI
                    ts1 = mean_values[i]
                    ts2 = mean_values[j]
                    
                    # Find indices where both time series have valid values
                    valid_indices = ~np.isnan(ts1) & ~np.isnan(ts2)
                    
                    if np.sum(valid_indices) > 1:
                        # Calculate correlation
                        corr, _ = stats.pearsonr(ts1[valid_indices], ts2[valid_indices])
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = np.nan
            
            return {
                'roi_ids': roi_ids,
                'correlation_matrix': corr_matrix.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def export_results_to_csv(self, results, file_path):
        """Export analysis results to CSV file."""
        self.logger.info(f"Exporting results to {file_path}")
        
        try:
            if not results:
                self.logger.error("No results to export")
                return False
            
            # Determine type of results
            if 'metrics' in results and 'time_points' in results:
                # Time series analysis results
                data = []
                
                # Add time points as first column
                row_data = {'Time Point': results['time_points']}
                
                # Add metrics for each ROI
                for roi_id, metrics in results['metrics'].items():
                    for metric_name, values in metrics.items():
                        column_name = f"{roi_id}_{metric_name}"
                        row_data[column_name] = values
                
                # Create DataFrame
                df = pd.DataFrame(row_data)
                
                # Save to CSV
                df.to_csv(file_path, index=False)
                
                return True
                
            elif 'frame' in results and 'mean' in results:
                # Basic statistics results
                df = pd.DataFrame(results)
                df.to_csv(file_path, index=False)
                
                return True
                
            else:
                self.logger.error("Unsupported results format for CSV export")
                return False
            
        except Exception as e:
            self.logger.error(f"Error exporting results to CSV: {e}")
            return False
    
    def create_time_series_plot(self, time_series, roi_ids=None, metric='mean'):
        """Create a matplotlib figure with time series plot."""
        self.logger.info("Creating time series plot")
        
        try:
            if not time_series:
                self.logger.error("No time series data available for plotting")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get time points
            time_points = time_series['time']
            
            # Plot time series for each ROI
            if 'region_ids' in time_series and roi_ids is None:
                # Plot all ROIs
                for i, roi_id in enumerate(time_series['region_ids']):
                    if metric in time_series and i < len(time_series[metric]):
                        values = time_series[metric][i]
                        ax.plot(time_points, values, label=roi_id)
                    
            elif 'region_ids' in time_series and roi_ids is not None:
                # Plot specific ROIs
                for roi_id in roi_ids:
                    if roi_id in time_series['region_ids']:
                        idx = time_series['region_ids'].index(roi_id)
                        if metric in time_series and idx < len(time_series[metric]):
                            values = time_series[metric][idx]
                            ax.plot(time_points, values, label=roi_id)
                    
            else:
                # Plot single time series
                if metric in time_series:
                    values = time_series[metric]
                    ax.plot(time_points, values, label=metric)
            
            # Add labels and legend
            ax.set_xlabel('Frame')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Time')
            ax.legend()
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating time series plot: {e}")
            return None
    
    def create_correlation_heatmap(self, correlation_data):
        """Create a matplotlib figure with correlation heatmap."""
        self.logger.info("Creating correlation heatmap")
        
        try:
            if not correlation_data or 'correlation_matrix' not in correlation_data:
                self.logger.error("No correlation data available for plotting")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get data
            corr_matrix = np.array(correlation_data['correlation_matrix'])
            roi_ids = correlation_data['roi_ids']
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Correlation')
            
            # Add labels
            ax.set_xticks(np.arange(len(roi_ids)))
            ax.set_yticks(np.arange(len(roi_ids)))
            ax.set_xticklabels(roi_ids, rotation=45, ha='right')
            ax.set_yticklabels(roi_ids)
            
            # Add title
            ax.set_title('ROI Correlation Matrix')
            
            # Adjust layout
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            return None

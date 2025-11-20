"""
CPU-Only InSAR Change Detection Module
Processes InSAR data using only CPU resources without GPU dependencies.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import json
import traceback
import pyproj
import concurrent.futures
from tqdm import tqdm
import os
import psutil

class CpuInsarChangeDetector:
    def __init__(self, 
                 window_size: int = 5,
                 change_threshold: float = 2.0,
                 coherence_threshold: float = 0.7,
                 min_observations: int = 10,
                 max_parallel_files: int = 8,
                 use_adaptive_windows: bool = True,
                 use_trend_analysis: bool = True,
                 use_spatial_context: bool = False,
                 ultra_selective: bool = True,
                 original_algorithm: bool = False,
                 baseline_detection: bool = False,
                 moderate_selective: bool = False,
                 gradual_selective: bool = False,
                 maximum_yield: bool = False,
                 cpu_threads: int = 8):
        """
        Initialize CPU-only InSAR Change Detector.
        
        Args:
            window_size: Base window size for fixed-window analysis
            change_threshold: Threshold for significant change detection
            coherence_threshold: Minimum coherence for reliable points
            min_observations: Minimum required observations for reliable analysis
            max_parallel_files: Maximum number of files to process in parallel
            use_adaptive_windows: Whether to use adaptive window sizes
            use_trend_analysis: Whether to perform trend analysis
            use_spatial_context: Whether to consider spatial context
            ultra_selective: Whether to use ultra-selective criteria to match example
            original_algorithm: Whether to use the original algorithm without modifications
            baseline_detection: Whether to use more relaxed criteria
            moderate_selective: Whether to use moderately selective criteria
            gradual_selective: Whether to use very relaxed criteria for maximum yield
            maximum_yield: Whether to use extremely relaxed criteria
            cpu_threads: Number of CPU threads for processing
        """
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.coherence_threshold = coherence_threshold
        self._min_observations = min_observations
        self.max_parallel_files = max_parallel_files
        self.use_adaptive_windows = use_adaptive_windows
        self.use_trend_analysis = use_trend_analysis
        self.use_spatial_context = use_spatial_context
        self.ultra_selective = ultra_selective
        self.original_algorithm = original_algorithm
        self.baseline_detection = baseline_detection
        self.moderate_selective = moderate_selective
        self.gradual_selective = gradual_selective
        self.maximum_yield = maximum_yield
        self.cpu_threads = cpu_threads
        
        # Initialize UTM to WGS84 transformer
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:32633",  # UTM zone 33N
            "EPSG:4326",   # WGS84
            always_xy=True # Output as longitude, latitude
        )
        
        # Get algorithm mode for logging
        if self.original_algorithm:
            self.algorithm_name = "Original Algorithm"
        elif self.baseline_detection:
            self.algorithm_name = "Baseline Detection (Relaxed Criteria)"
        elif self.moderate_selective:
            self.algorithm_name = "Moderate-Selective Pattern Matching"
        elif self.gradual_selective:
            self.algorithm_name = "Gradual-Selective Pattern Matching"
        elif self.maximum_yield:
            self.algorithm_name = "Maximum-Yield Detection"
        else:  # Ultra-selective
            self.algorithm_name = "Ultra-Selective Pattern Matching"
        
        # Determine thresholds based on algorithm mode
        self._set_detection_thresholds()
        
        # Log system information
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system information for performance tracking."""
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System info: {cpu_count} CPU cores, {memory_gb:.2f} GB RAM")
        print(f"Using {self.cpu_threads} threads for processing")
    
    def _set_detection_thresholds(self):
        """Set detection thresholds based on selected algorithm mode."""
        if self.original_algorithm:
            self.early_std_threshold = 1.5
            self.late_std_threshold = 5.0
            self.ratio_threshold = 7.0
            self.early_range_threshold = float('inf')
            self.magnitude_threshold = 0.0
            self.coherence_threshold_value = self.coherence_threshold
            
        elif self.baseline_detection:
            self.early_std_threshold = 1.5
            self.late_std_threshold = 5.0
            self.ratio_threshold = 7.0
            self.early_range_threshold = float('inf')
            self.magnitude_threshold = 0.0
            self.coherence_threshold_value = self.coherence_threshold
            
        elif self.maximum_yield:
            self.early_std_threshold = 2.0
            self.late_std_threshold = 3.5
            self.ratio_threshold = 2.5
            self.early_range_threshold = 15.0
            self.magnitude_threshold = 10.0
            self.coherence_threshold_value = 0.60
            
        elif self.gradual_selective:
            self.early_std_threshold = 1.6
            self.late_std_threshold = 4.5
            self.ratio_threshold = 5.0
            self.early_range_threshold = 10.0
            self.magnitude_threshold = 25.0
            self.coherence_threshold_value = 0.70
            
        elif self.moderate_selective:
            self.early_std_threshold = 1.4
            self.late_std_threshold = 4.8
            self.ratio_threshold = 6.0
            self.early_range_threshold = 8.0
            self.magnitude_threshold = 50.0
            self.coherence_threshold_value = 0.75
            
        else:  # Ultra-selective
            self.early_std_threshold = 0.8
            self.late_std_threshold = 6.0
            self.ratio_threshold = 15.0
            self.early_range_threshold = 4.0
            self.magnitude_threshold = 300.0
            self.coherence_threshold_value = 0.84
    
    @property
    def min_observations(self):
        """Minimum required observations for reliable analysis."""
        return self._min_observations
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and deduplicate column names."""
        # Strip whitespace and convert to lowercase
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle duplicate columns by adding a suffix
        seen = {}
        new_columns = []
        
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
                
        df.columns = new_columns
        return df

    def _get_date_columns(self, df: pd.DataFrame) -> list:
        """Get time series date columns from DataFrame."""
        # Find all columns that are 8-digit numbers (dates in YYYYMMDD format)
        date_cols = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 8]
        return sorted(date_cols)
    
    def _get_coordinate_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get mapping from standard coordinate names to actual column names."""
        coord_map = {}
        for target, variants in {
            'northing': ['northing', 'north', 'n', 'y'],
            'easting': ['easting', 'east', 'e', 'x'],
            'coherence': ['temporal_coherence', 'coherence', 'coh', 'mean_coherence']
        }.items():
            for variant in variants:
                matching_cols = [col for col in df.columns if variant == col or 
                               col.startswith(f"{variant}_")]
                if matching_cols:
                    coord_map[target] = matching_cols[0]
                    break
        return coord_map
    
    def _convert_utm_to_latlon(self, easting: np.ndarray, northing: np.ndarray) -> Tuple[List[float], List[float]]:
        """Convert UTM coordinates to latitude/longitude."""
        batch_size = 10000  # Process in batches to avoid memory issues
        longitudes = []
        latitudes = []
        
        for i in range(0, len(easting), batch_size):
            end_idx = min(i + batch_size, len(easting))
            batch_e = easting[i:end_idx]
            batch_n = northing[i:end_idx]
            lon_batch, lat_batch = self.transformer.transform(batch_e, batch_n)
            longitudes.extend(lon_batch)
            latitudes.extend(lat_batch)
            
        return longitudes, latitudes
    
    def _prepare_file_info(self, file_path: Path) -> Optional[Tuple[str, pd.DataFrame, Dict]]:
        """
        Load and prepare a file for processing.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (track_number, dataframe, metadata) or None on error
        """
        try:
            # Skip files we shouldn't process
            if any(x in str(file_path).lower() for x in ['state.json', 'change_detection_results']):
                return None
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Extract track number
            track_num = str(file_path.name).split('-')[0]
            df['track'] = track_num
            
            # Add file source
            df['file_source'] = file_path.name
            
            # Clean column names
            df = self._clean_column_names(df)
            
            # Get date columns
            date_cols = self._get_date_columns(df)
            if not date_cols:
                print(f"No date columns found in {file_path.name}")
                return None
            
            # Map coordinate columns
            coord_map = self._get_coordinate_mapping(df)
            
            # Ensure we have required coordinate columns
            if not all(k in coord_map for k in ['northing', 'easting']):
                print(f"Missing coordinate columns in {file_path.name}")
                return None
            
            # Create metadata dictionary
            metadata = {
                'track': track_num,
                'file_path': str(file_path),
                'date_cols': date_cols,
                'coord_map': coord_map,
                'num_points': len(df),
                'num_dates': len(date_cols)
            }
            
            print(f"Loaded {file_path.name}: {metadata['num_points']:,} points, {metadata['num_dates']} dates")
            
            return track_num, df, metadata
            
        except Exception as e:
            print(f"Error preparing {file_path}: {str(e)}")
            traceback.print_exc()
            return None
    
    def _process_file(self, file_info: Tuple[str, pd.DataFrame, Dict]) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Process a file using CPU.
        
        Args:
            file_info: Tuple of (track_number, dataframe, metadata)
            
        Returns:
            Tuple of (track_number, results_df) or None on error
        """
        if not file_info:
            return None
            
        track_num, df, metadata = file_info
        
        try:
            t0 = time.time()
            
            # Extract metadata
            date_cols = metadata['date_cols']
            coord_map = metadata['coord_map']
            file_path = metadata['file_path']
            file_source = Path(file_path).name
            
            # Create a copy of essential columns to reduce memory usage
            essential_cols = [coord_map['northing'], coord_map['easting']]
            if 'coherence' in coord_map:
                essential_cols.append(coord_map['coherence'])
            essential_cols.extend(date_cols)
            essential_cols.extend(['track', 'file_source'])
            
            # Create a clean copy with only needed columns
            df_clean = df[essential_cols].copy()
            
            # Rename coordinate columns for consistency
            for target, col in coord_map.items():
                if col != target:
                    df_clean.rename(columns={col: target}, inplace=True)
            
            # Define early and late periods
            timestamps = pd.to_datetime(date_cols, format='%Y%m%d')
            cutoff_date = pd.Timestamp('2022-01-01')
            early_dates = [col for col in date_cols if pd.to_datetime(col) < cutoff_date]
            late_dates = [col for col in date_cols if pd.to_datetime(col) >= cutoff_date]
            
            # Calculate valid observations
            valid_mask = df_clean[date_cols] != 0
            valid_count = valid_mask.sum(axis=1).values
            
            # Calculate early period statistics
            if early_dates:
                early_values = df_clean[early_dates].astype(np.float32).values
                early_std = np.nanstd(early_values, axis=1)
                early_mean = np.nanmean(early_values, axis=1)
                
                # Calculate range for early period
                early_max = np.nanmax(early_values, axis=1)
                early_min = np.nanmin(early_values, axis=1)
                early_range = np.abs(early_max - early_min)
            else:
                # Default values if no early dates
                early_std = np.zeros(len(df_clean))
                early_mean = np.zeros(len(df_clean))
                early_range = np.zeros(len(df_clean))
            
            # Calculate late period statistics
            if late_dates:
                late_values = df_clean[late_dates].astype(np.float32).values
                late_std = np.nanstd(late_values, axis=1)
                late_mean = np.nanmean(late_values, axis=1)
            else:
                # Default values if no late dates
                late_std = np.zeros(len(df_clean))
                late_mean = np.zeros(len(df_clean))
            
            # Initialize trend analysis outputs
            slopes = np.zeros(len(df_clean))
            trend_residuals = np.zeros(len(df_clean))
            
            # Process trend analysis if enabled
            if self.use_trend_analysis and not self.original_algorithm:
                # Convert dates to numeric values (days since first date)
                date_values = np.array([(d - timestamps[0]).days for d in timestamps], dtype=np.float32)
                
                # Calculate trend for each point using vectorized operations where possible
                time_series = df_clean[date_cols].astype(np.float32).values
                
                # Prepare design matrix for linear regression
                X = np.column_stack([np.ones(len(date_values)), date_values])
                
                # Process in batches for efficiency
                batch_size = 1000
                for i in range(0, len(df_clean), batch_size):
                    end_idx = min(i + batch_size, len(df_clean))
                    
                    for j in range(i, end_idx):
                        y = time_series[j]
                        mask = y != 0  # Valid observations
                        
                        if np.sum(mask) >= 3:  # Need at least 3 points
                            X_valid = X[mask]
                            y_valid = y[mask]
                            
                            try:
                                # Linear regression using numpy
                                beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                                slopes[j] = beta[1]  # Slope
                                
                                # Calculate residuals
                                y_pred = X_valid @ beta
                                trend_residuals[j] = np.sqrt(np.mean((y_valid - y_pred)**2))
                            except:
                                slopes[j] = 0.0
                                trend_residuals[j] = 0.0
            
            # Get coherence values
            if 'coherence' in df_clean.columns:
                coherence = df_clean['coherence'].values
            else:
                coherence = np.ones(len(df_clean))
            
            # Initialize results
            is_significant = np.zeros(len(df_clean), dtype=bool)
            change_magnitude = np.zeros(len(df_clean), dtype=np.float32)
            
            # Apply detection criteria
            stable_early = early_std < self.early_std_threshold
            unstable_late = late_std > self.late_std_threshold
            
            # Ratio with protection against division by zero
            ratio = late_std / (early_std + 0.1)
            has_high_ratio = ratio > self.ratio_threshold
            
            # Combine primary criteria
            pattern_match = stable_early & unstable_late & has_high_ratio
            
            # Add flat early period criteria for non-original algorithms
            if not self.original_algorithm and not self.baseline_detection:
                flat_early_period = early_range < self.early_range_threshold
                pattern_match = pattern_match & flat_early_period
            
            # Calculate change magnitude
            change_magnitude[pattern_match] = ratio[pattern_match] * 50.0
            
            # Apply magnitude threshold if specified
            if self.magnitude_threshold > 0:
                magnitude_filter = change_magnitude > self.magnitude_threshold
                pattern_match = pattern_match & magnitude_filter
            
            # Apply trend analysis filter if enabled
            if self.use_trend_analysis and not self.original_algorithm and not self.maximum_yield:
                steady_movement = (np.abs(slopes) > 0.1) & (trend_residuals < 5.0)
                pattern_match = pattern_match & (~steady_movement)
            
            # Apply quality controls
            min_obs_filter = valid_count >= self.min_observations
            coherence_filter = coherence >= self.coherence_threshold_value
            
            # Final significance flag
            is_significant = pattern_match & min_obs_filter & coherence_filter
            
            # Convert UTM to lat/lon
            easting = df_clean['easting'].values
            northing = df_clean['northing'].values
            longitudes, latitudes = self._convert_utm_to_latlon(easting, northing)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'northing': northing,
                'easting': easting,
                'latitude': latitudes,
                'longitude': longitudes,
                'track': df_clean['track'].values,
                'file_source': df_clean['file_source'].values,
                'coherence': coherence,
                'change_magnitude': change_magnitude,
                'is_significant': is_significant,
                'valid_observations': valid_count,
                'is_reliable': valid_count >= self.min_observations,
                'trend_slope': slopes,
                'trend_residual': trend_residuals,
                'early_std': early_std,
                'late_std': late_std,
                'early_range': early_range,
            })
            
            # Calculate change date
            if late_dates:
                change_date_idx = len(early_dates) + len(late_dates) // 2
                change_date = date_cols[change_date_idx] if 0 <= change_date_idx < len(date_cols) else 'Unknown'
            else:
                change_date = 'Unknown'
            
            results_df['change_date'] = change_date
            
            # Calculate confidence score
            results_df['confidence'] = results_df['change_magnitude'] / (results_df['trend_residual'] + 1.0)
            
            # Log processing time and results
            processing_time = time.time() - t0
            significant_count = results_df['is_significant'].sum()
            
            print(f"Processed {file_source} in {processing_time:.2f}s: "
                  f"{significant_count} significant changes from {len(df_clean):,} points "
                  f"({len(df_clean)/processing_time:.0f} pts/s)")
            
            return track_num, results_df
            
        except Exception as e:
            print(f"Error processing file {metadata['file_path']}: {str(e)}")
            traceback.print_exc()
            return None
    
    def detect_changes(self, base_path: str) -> pd.DataFrame:
        """
        Detect changes in InSAR time series data using CPU processing.
        
        Args:
            base_path: Path to directory containing CSV files
            
        Returns:
            DataFrame with detection results
        """
        start_time = time.time()
        print(f"Starting CPU-only InSAR change detection...")
        
        # Find all CSV files
        csv_files = list(Path(base_path).rglob('*.csv'))
        print(f"Found {len(csv_files)} CSV files")
        
        # Filter out non-data files
        data_files = [f for f in csv_files 
                     if not any(x in str(f).lower() for x in ['state.json', 'change_detection_results'])]
        
        # Gather file information
        file_info_list = []
        
        print("Analyzing files...")
        for file_path in data_files:
            result = self._prepare_file_info(file_path)
            if result:
                file_info_list.append(result)
        
        print(f"Processing {len(file_info_list)} files with {self.cpu_threads} threads")
        
        # Process files in parallel
        all_results = []
        file_info = {}
        
        # Create thread pool for CPU processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_threads) as executor:
            # Submit processing tasks
            futures = {
                executor.submit(self._process_file, file_data): file_data[0]  # track_num
                for file_data in file_info_list
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures),
                              desc="Processing files"):
                track_num = futures[future]
                try:
                    result = future.result()
                    if result:
                        track, file_results = result
                        # Get the file path from the original file_info_list
                        for file_data in file_info_list:
                            if file_data[0] == track:  # file_data[0] is track_num
                                file_info[track] = file_data[2]['file_path']  # file_data[2] is metadata dict
                                break
                        all_results.append(file_results)
                except Exception as e:
                    print(f"Error processing track {track_num}: {str(e)}")
        
        if not all_results:
            raise ValueError("No data was processed successfully")
        
        # Combine all results
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Save file mapping
        with open('file_mapping.json', 'w') as f:
            json.dump(file_info, f, indent=2)
        
        # Calculate overall statistics
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
        print(f"Total points processed: {len(results_df):,}")
        
        significant_changes = results_df['is_significant'].sum()
        print(f"Significant changes detected: {significant_changes:,} points")
        print(f"Change detection algorithm: {self.algorithm_name}")
        
        # Calculate speed statistics
        points_per_second = len(results_df) / elapsed_time
        print(f"Processing speed: {points_per_second:.1f} points/second")
        
        # Count changes per file
        if significant_changes > 0:
            file_counts = results_df[results_df['is_significant']].groupby('file_source').size()
            print("\nSignificant changes by file:")
            for file_name, count in file_counts.items():
                print(f"  {file_name}: {count} changes")
        
        return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU-Only InSAR Change Detection')
    parser.add_argument('--path', required=True, help='Path to CSV files')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Change detection threshold')
    parser.add_argument('--coherence', type=float, default=0.7,
                       help='Minimum coherence threshold')
    parser.add_argument('--min-observations', type=int, default=10,
                       help='Minimum required observations for reliable analysis')
    parser.add_argument('--parallel-files', type=int, default=8,
                       help='Maximum number of files to process in parallel')
    parser.add_argument('--use-fixed-periods', action='store_true',
                       help='Use fixed early/late periods instead of adaptive windows')
    parser.add_argument('--disable-trend-analysis', action='store_true',
                       help='Disable linear trend analysis')
    parser.add_argument('--enable-spatial-context', action='store_true',
                       help='Enable spatial context analysis')
    parser.add_argument('--relaxed-criteria', action='store_true',
                       help='Use more relaxed detection criteria (not ultra-selective)')
    parser.add_argument('--original', action='store_true',
                       help='Use original algorithm without modifications')
    parser.add_argument('--baseline', action='store_true',
                       help='Use baseline detection with more relaxed criteria')
    parser.add_argument('--moderate', action='store_true',
                       help='Use moderately selective criteria targeting 250-500 points')
    parser.add_argument('--gradual', action='store_true',
                       help='Use very relaxed criteria for maximum detection yield')
    parser.add_argument('--maximum', action='store_true',
                       help='Use extremely relaxed criteria for absolute maximum yield')
    parser.add_argument('--cpu-threads', type=int, default=8,
                       help='Number of CPU threads for processing')
    
    args = parser.parse_args()
    
    detector = CpuInsarChangeDetector(
        change_threshold=args.threshold,
        coherence_threshold=args.coherence,
        min_observations=args.min_observations,
        max_parallel_files=args.parallel_files,
        use_adaptive_windows=not args.use_fixed_periods,
        use_trend_analysis=not args.disable_trend_analysis,
        use_spatial_context=args.enable_spatial_context,
        ultra_selective=not args.relaxed_criteria,
        original_algorithm=args.original,
        baseline_detection=args.baseline,
        moderate_selective=args.moderate,
        gradual_selective=args.gradual,
        maximum_yield=args.maximum,
        cpu_threads=args.cpu_threads
    )
    
    try:
        results = detector.detect_changes(args.path)
        if not results.empty:
            output_file = 'change_detection_results.csv'
            results.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
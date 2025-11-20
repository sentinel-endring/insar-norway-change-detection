"""
InSAR Time Series Visualization Module (CPU-Only Version)
Generates time series plots for points of interest from change detection results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time
import json
import argparse

class InsarVisualizerCpu:
    def __init__(self, file_paths: dict, results_file: str):
        """
        Initialize the InSAR visualizer.
        
        Args:
            file_paths: Dictionary mapping track numbers to file paths
            results_file: Path to change detection results CSV
        """
        self.file_paths = file_paths
        self.results_df = pd.read_csv(results_file)
        # Filter only significant changes
        self.results_df = self.results_df[self.results_df['is_significant']]
        print(f"Loaded {len(self.results_df)} significant changes from results file")
    
    def _find_closest_points_cpu(self, 
                                points_df: pd.DataFrame,
                                target_coords: tuple,
                                max_distance: float = 0.5) -> pd.DataFrame:
        """Find points closest to target coordinates using CPU."""
        # Convert coordinates to numpy arrays
        coords = points_df[['northing', 'easting']].values
        target = np.array([float(target_coords[0]), float(target_coords[1])]).reshape(1, 2)
        
        # Calculate distances
        distances = np.sqrt(np.sum((coords - target) ** 2, axis=1))
        
        # Find points within max distance
        valid_points = np.where(distances <= max_distance)[0]
        
        if len(valid_points) > 0:
            closest_points = points_df.iloc[valid_points].copy()
            closest_points['distance'] = distances[valid_points]
            return closest_points.sort_values('distance')
        
        return pd.DataFrame()
    
    def _get_time_series_columns(self, df: pd.DataFrame) -> list:
        """Extract time series columns (dates) from DataFrame."""
        date_cols = []
        for col in df.columns:
            # Look for YYYYMMDD pattern
            if str(col).isdigit() and len(str(col)) == 8:
                try:
                    datetime.strptime(str(col), '%Y%m%d')
                    date_cols.append(col)
                except ValueError:
                    continue
        return sorted(date_cols)
    
    def _create_time_series_plot(self,
                               point_data: pd.DataFrame,
                               file_info: str,
                               point_info: dict,
                               output_file: str):
        """Create time series plot for a single point."""
        plt.figure(figsize=(15, 8))
        
        # Get time series columns
        date_cols = self._get_time_series_columns(point_data)
        if not date_cols:
            print(f"No time series data found in {file_info}")
            plt.close()
            return
        
        # Convert dates to datetime and values to numeric
        dates = [datetime.strptime(str(col), '%Y%m%d') for col in date_cols]
        values = point_data[date_cols].iloc[0].astype(float).values
        
        # Create main plot
        plt.subplot(111)
        plt.plot(dates, values, 'r.', markersize=4, label='Measurements')
        plt.plot(dates, values, 'b-', linewidth=0.5, alpha=0.5)
        
        # Add reference line at y=0
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        title = f"Track Information: {file_info}\n"
        title += "SIGNIFICANT CHANGE DETECTED\n"
        title += f"Change Magnitude: {point_info['change_magnitude']:.3f}"
        plt.title(title)
        
        plt.xlabel('Measurement date')
        plt.ylabel('Displacement (mm)')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add point information
        info_text = [
            f"File: {Path(file_info).name}",
            f"Coordinates: N={point_info['northing']:.1f}, E={point_info['easting']:.1f}",
            f"Change Magnitude: {point_info['change_magnitude']:.2f}",
            f"Coherence: {point_info['coherence']:.3f}"
        ]
        
        plt.text(0.02, 0.98, '\n'.join(info_text),
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add track direction indicator
        ax2 = plt.axes([0.85, 0.7, 0.1, 0.1])
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False)
        ax2.add_patch(circle)
        
        # Add cardinal directions
        ax2.text(0.5, 1.1, 'N', ha='center')
        ax2.text(0.5, -0.1, 'S', ha='center')
        ax2.text(1.1, 0.5, 'E', va='center')
        ax2.text(-0.1, 0.5, 'W', va='center')
        
        # Determine track angle based on track type (Ascending or Descending)
        track_type = 'A' if 'A' in file_info else 'D'
        track_angle = 346.13 if track_type == 'A' else 193.87
        
        # Add line indicating track direction
        dx = 0.3 * np.cos(np.radians(track_angle))
        dy = 0.3 * np.sin(np.radians(track_angle))
        ax2.arrow(0.5-dx, 0.5-dy, 2*dx, 2*dy, 
                 head_width=0.1, 
                 color='blue',
                 length_includes_head=True)
        
        ax2.set_xlim(-0.2, 1.2)
        ax2.set_ylim(-0.2, 1.2)
        ax2.axis('off')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_visualizations(self, 
                              num_points: int = 5,
                              output_dir: str = 'visualizations',
                              selection: str = 'highest'):
        """Generate visualizations for points based on change magnitude.
        
        Args:
            num_points: Number of points to visualize per category
            output_dir: Directory to save visualization files
            selection: One of 'highest', 'lowest', 'middle', 'all' or 'mixed'
                      'mixed' will generate visualizations for highest, middle and lowest points
        """
        print("Generating visualizations...")
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare points based on selection criteria
        if selection == 'mixed':
            # Get highest points
            highest_points = self.results_df.nlargest(num_points, 'change_magnitude')
            
            # Get lowest points
            lowest_points = self.results_df.nsmallest(num_points, 'change_magnitude')
            
            # Get middle points
            middle_idx = len(self.results_df) // 2
            middle_points = self.results_df.sort_values('change_magnitude').iloc[middle_idx-num_points//2:middle_idx+num_points//2]
            
            points_to_process = pd.concat([highest_points, middle_points, lowest_points])
            categories = ['highest'] * num_points + ['middle'] * num_points + ['lowest'] * num_points
            
        elif selection == 'middle':
            middle_idx = len(self.results_df) // 2
            points_to_process = self.results_df.sort_values('change_magnitude').iloc[middle_idx-num_points//2:middle_idx+num_points//2]
            categories = ['middle'] * len(points_to_process)
            
        elif selection == 'lowest':
            points_to_process = self.results_df.nsmallest(num_points, 'change_magnitude')
            categories = ['lowest'] * len(points_to_process)
            
        elif selection == 'highest':
            points_to_process = self.results_df.nlargest(num_points, 'change_magnitude')
            categories = ['highest'] * len(points_to_process)
            
        else:  # 'all'
            points_to_process = self.results_df
            categories = ['all'] * len(self.results_df)
        
        # Process each point
        for (idx, row), category in zip(points_to_process.iterrows(), categories):
            print(f"\nProcessing point {idx} with change magnitude {row['change_magnitude']:.2f}")
            coords = (row['northing'], row['easting'])
            
            # Process track
            track = str(row['track'])
            if track in self.file_paths:
                file_path = self.file_paths[track]
                print(f"Processing file: {file_path}")
                
                # Ensure file_path is a full path
                file_path = Path(file_path)
                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue
                
                try:
                    # Load data
                    track_df = pd.read_csv(file_path, skipinitialspace=True)
                    
                    # Find closest point
                    closest_points = self._find_closest_points_cpu(
                        track_df, coords)
                    
                    if not closest_points.empty:
                        output_file = output_path / f"{category}_change_point_{idx}_track_{track}.png"
                        
                        # Create visualization
                        self._create_time_series_plot(
                            closest_points,
                            str(file_path),
                            row,
                            output_file
                        )
                        print(f"Created visualization: {output_file}")
                    else:
                        print(f"No close points found in {file_path}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            else:
                print(f"No file path found for track {track}")
        
        print(f"\nVisualization generation completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InSAR Time Series Visualization (CPU-Only)')
    parser.add_argument('--results', required=True, help='Path to change detection results CSV')
    parser.add_argument('--num-points', type=int, default=5,
                       help='Number of points to visualize')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Directory to save visualization files')
    parser.add_argument('--selection', default='highest', 
                       choices=['highest', 'lowest', 'middle', 'mixed', 'all'],
                       help='Which points to visualize based on change magnitude')
    
    args = parser.parse_args()
    
    try:
        # Load file mapping from JSON
        with open('file_mapping.json', 'r') as f:
            file_paths = json.load(f)
            print("\nLoaded file mapping:")
            for track, path in file_paths.items():
                print(f"Track {track}: {path}")
        
        visualizer = InsarVisualizerCpu(file_paths, args.results)
        visualizer.generate_visualizations(
            num_points=args.num_points,
            output_dir=args.output_dir,
            selection=args.selection
        )
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        raise
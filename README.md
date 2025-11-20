# InSAR-Norway-Change-Detection

A Python-based toolkit for querying, processing, and analyzing InSAR (Interferometric Synthetic Aperture Radar) data from the Norwegian Ground Motion Service. This tool specializes in detecting and analyzing ground motion changes using Sentinel-1 satellite data.

## Features

- Query multiple Sentinel-1 datasets from InSAR Norway (insar.ngu.no) for various time periods.
- Advanced CPU-based change detection analysis with multiple algorithms (e.g., ultra-selective, baseline, flexible-period).
- Interactive GIS Map Viewer to visualize significant change points on a map with historical data and GeoTIFF overlays.
- Time-series plotting for detailed analysis of specific points.
- Support for both ascending and descending satellite tracks.
- Statistical analysis of temporal changes and coherence.
- Legacy GPU-accelerated code is available in the `gpu_legacy` directory.

## Prerequisites

- Linux distributions with `glibc>=2.28`
- Python 3.12
- Conda package manager

For GPU-accelerated processing, please see the `gpu_legacy` directory and its specific requirements.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://gitlab.terratera.net/copernicus-samarbeid/insar-norway-change-detection.git
    cd insar-norway-change-detection
    ```

2.  Create and activate the Conda environment:
    ```bash
    # For CPU-Only environment
    conda env create -f environment.yml
    conda activate insar-cpu-env
    ```
    **Note:** The environment name is defined in the `environment.yml` file (default: `insar-cpu-env`).

## Usage

The workflow is divided into three main steps: Querying, Processing, and Visualization.

### 1. Querying Data

The `insar-query.py` script allows you to download data from the InSAR Norway portal.

```bash
python insar-query.py --path /path/to/output --bbox=lon1,lat1,lon2,lat2 --period 2019-2023
```

**Example:**

```bash
python insar-query.py --path /data/insar-norway-change-detection --bbox=7.986722817,58.149864623,7.991700669,58.146240529
```

**Key Options:**

-   `--path`: Directory to store query results.
-   `--bbox`: Bounding box coordinates (longitude,latitude,longitude,latitude).
-   `--period`: Time period for Sentinel-1 datasets. Options include `2019-2023`, `2020-2024`, `latest`, `all`. Default is `2019-2023`.
-   `--list-datasets`: List all available datasets on the server.

### 2. Processing Data

The `hybrid_detector.py` script processes the downloaded data to find significant changes. It offers several detection algorithms.

```bash
python hybrid_detector.py --path /path/to/downloaded_data --moderate
```

This script will analyze the CSV files in the `--path` directory and generate `change_detection_results.csv` and `file_mapping.json`.

**Detection Algorithms:**

Choose one of the following flags to select the detection algorithm.

-   `--ultra-selective`: (Default) Very strict criteria to find only the most dramatic changes.
-   `--moderate`: A balanced and moderately selective algorithm.
-   `--gradual`: Relaxed criteria to detect more subtle and gradual changes.
-   `--maximum`: Extremely relaxed criteria for maximum detection yield.
-   `--baseline`: A relaxed criteria set for baseline detection.
-   `--original`: The original, legacy algorithm.

**Other Options:**

-   `--cpu-threads`: Number of CPU threads to use.
-   `--coherence`: Minimum coherence threshold (default: 0.7).
-   `--fixed-2022-cutoff`: Use a fixed date cutoff (2022-01-01) for change detection instead of the default flexible period detection.

### 3. Visualizing Results

There are two ways to visualize the results: an interactive map viewer and static time-series plots.

#### Interactive GIS Map Viewer

The `gis_map_viewer.py` script launches a web-based map to explore significant change points interactively.

**Prerequisites for the Viewer:**

The viewer requires the following files and folders to be present in the root directory:

-   `change_detection_results.csv`: The output from the processing step.
-   `binary_detection.tif` (Optional): A GeoTIFF file to be overlaid on the map.
-   `Basisdata_..._FGDB/` (Optional): Folders containing historical data in FileGDB format.

**Running the Viewer:**

```bash
python gis_map_viewer.py
```

This will start a local web server and open the map in your default browser. You can toggle layers, click on points for detailed information, and explore the detected changes in their geographical context.

#### Time-Series Plots

The `insar-visualizer.py` script generates static plots for a selection of points from the results, allowing for detailed inspection of their time-series data.

**Usage:**

```bash
python insar-visualizer.py --results change_detection_results.csv --num-points 5 --selection mixed
```

**Key Options:**

-   `--results`: Path to the `change_detection_results.csv` file.
-   `--num-points`: Number of points to visualize for each category.
-   `--selection`: Which points to visualize based on change magnitude.
    -   `highest`: Points with the largest change magnitude.
    -   `lowest`: Points with the smallest change magnitude.
    -   `middle`: Points with a medium change magnitude.
    -   `mixed`: A mix of highest, lowest, and middle points.
    -   `all`: Visualize all significant points.
-   `--output-dir`: Directory to save the generated plots (default: `visualizations`).

## Output

-   `change_detection_results.csv`: A CSV file containing all detected changes with detailed metrics (coordinates, magnitude, coherence, etc.).
-   `file_mapping.json`: A JSON file that maps internal IDs to the original data filenames.
-   `visualizations/`: A directory containing the generated time-series plots (if `insar-visualizer.py` is used).

## GPU Legacy Code

The `gpu_legacy` directory contains older, GPU-accelerated versions of the processing and visualization scripts. These are no longer actively maintained but are available for users with compatible NVIDIA hardware. Please refer to the `gpu_legacy/README.md` for setup and usage instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work has been partly funded by the Norwegian Space Agency, contract number 74CO2504.
This work would not been possible without Norwegian Ground Motion Service (insar.ngu.no) for providing the InSAR data.

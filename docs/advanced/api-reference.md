# API Reference

This section provides detailed reference information for the main scripts and their command-line interfaces.

## Core Scripts

### `insar-query.py`

Query and download InSAR data from the Norwegian Ground Motion Service.

```bash
python insar-query.py [OPTIONS]
```

#### Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--path` | string | Output directory for downloaded data | Required |
| `--bbox` | string | Bounding box (lon1,lat1,lon2,lat2) | Required |
| `--period` | string | Time period: 2019-2023, 2020-2024, 2018-2022, latest, all | 2019-2023 |
| `--list-datasets` | flag | List available datasets without downloading | False |
| `--include-radarsat` | flag | Also include Radarsat-2 datasets | False |
| `--datasets-only` | flag | Only query Sentinel datasets, skip other types | False |
| `--host` | string | Host server to query | https://insar.ngu.no |
| `--cert` | string | Root certificate for verification | Empty |

#### Example Usage

```bash
# Basic query
python insar-query.py --path /data/output --bbox=10.7,59.9,10.8,60.0

# List available datasets
python insar-query.py --list-datasets

# Query specific time period
python insar-query.py --path /data --bbox=10.7,59.9,10.8,60.0 --period latest
```

### `hybrid_detector.py`

Process InSAR data to detect significant ground motion changes.

```bash
python hybrid_detector.py [OPTIONS] [ALGORITHM]
```

#### Algorithm Options (choose one)

| Algorithm | Description |
|-----------|-------------|
| `--ultra-selective` | Very strict criteria (default) |
| `--moderate` | Balanced detection |
| `--gradual` | Detect subtle changes |
| `--maximum` | Maximum sensitivity |
| `--baseline` | Baseline detection |
| `--original` | Legacy algorithm |

#### Processing Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--path` | string | Directory containing CSV data files | Required |
| `--cpu-threads` | int | Number of CPU threads for processing | 8 |
| `--coherence` | float | Minimum coherence threshold (0.0-1.0) | 0.7 |
| `--fixed-2022-cutoff` | flag | Use fixed 2022-01-01 cutoff instead of flexible periods | False |
| `--threshold` | float | Change detection threshold | 2.0 |
| `--min-observations` | int | Minimum required observations for analysis | 10 |
| `--parallel-files` | int | Maximum files to process in parallel | 8 |

#### Example Usage

```bash
# Standard processing
python hybrid_detector.py --path /data --moderate

# High-performance processing
python hybrid_detector.py --path /data --moderate --cpu-threads 8 --coherence 0.8

# Sensitive detection
python hybrid_detector.py --path /data --gradual --coherence 0.6
```

### `gis_map_viewer.py`

Launch interactive web-based map viewer for exploring results.

```bash
python gis_map_viewer.py
```

#### Notes

- No command-line options available
- Automatically runs on http://127.0.0.1:5000
- Opens browser automatically after startup
- Debug mode is enabled by default

#### Required Files

- `change_detection_results.csv`
- `binary_detection.tif` (optional)
- `Basisdata_*_FGDB/` directories (optional)

#### Example Usage

```bash
# Launch map viewer (only option)
python gis_map_viewer.py
```

### `insar-visualizer.py`

Generate time-series plots for detected changes.

```bash
python insar-visualizer.py [OPTIONS]
```

#### Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--results` | string | Path to change_detection_results.csv | Required |
| `--num-points` | int | Number of points per category | 5 |
| `--selection` | string | Point selection: highest, lowest, middle, mixed, all | mixed |
| `--output-dir` | string | Output directory for plots | visualizations |

#### Example Usage

```bash
# Basic visualization
python insar-visualizer.py --results change_detection_results.csv

# High-impact changes
python insar-visualizer.py --results results.csv --selection highest --num-points 10

# Comprehensive analysis
python insar-visualizer.py --results results.csv --selection all --output-dir comprehensive_plots
```

## CPU-Optimized Scripts

### `hybrid_detector-cpu.py`

CPU-optimized version of the change detection algorithm.

```bash
python hybrid_detector-cpu.py [OPTIONS]
```

Same parameters as `hybrid_detector.py` but optimized for CPU processing.

### `insar-visualizer-cpu.py`

CPU-optimized visualization script for large datasets.

```bash
python insar-visualizer-cpu.py [OPTIONS]
```

Same parameters as `insar-visualizer.py` with CPU optimizations.

## Output File Formats

### `change_detection_results.csv`

Main results file containing detected changes.

#### Columns

| Column | Type | Description |
|--------|------|-------------|
| `easting` | float | X coordinate in EPSG:3035 (meters) |
| `northing` | float | Y coordinate in EPSG:3035 (meters) |
| `longitude` | float | Point longitude in WGS84 (decimal degrees) |
| `latitude` | float | Point latitude in WGS84 (decimal degrees) |
| `track` | string | Satellite track identifier |
| `file_source` | string | Source CSV file name |
| `coherence` | float | Temporal coherence value (0-1) |
| `change_magnitude` | float | Change magnitude (mm) |
| `is_significant` | boolean | Whether change is statistically significant |
| `valid_observations` | int | Number of valid observations |
| `confidence` | float | Detection confidence score |

!!! info "Coordinate System Details"
    The CSV contains coordinates in **both** coordinate systems:
    
    - **EPSG:3035 (ETRS89-LAEA)**: `easting`, `northing` - Used by GIS viewer for accurate European analysis
    - **WGS84 (EPSG:4326)**: `latitude`, `longitude` - Converted for compatibility with standard mapping tools
    
    **EPSG:3035 is preferred** for analysis as it provides better accuracy for European data.
| `track_type` | string | Satellite track (ascending/descending) |

### `file_mapping.json`

Maps internal point IDs to original data filenames.

```json
{
  "point_001": "dataset1_ascending.csv",
  "point_002": "dataset1_descending.csv",
  "mapping_info": {
    "creation_date": "2024-01-15",
    "total_points": 1234
  }
}
```

## Return Codes

All scripts use standard return codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Network error |
| 5 | Processing error |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INSAR_DATA_PATH` | Default data directory | Current directory |
| `INSAR_THREADS` | Default CPU thread count | Auto-detect |
| `INSAR_COHERENCE` | Default coherence threshold | 0.7 |

## Configuration Files

### `environment.yml`

Conda environment specification for CPU processing.

### `environment-cpu.yml`

Alternative CPU-only environment file.

### `requirements.txt`

Pip requirements for manual installation.

## Python API

For programmatic access, key functions can be imported:

```python
from hybrid_detector import detect_changes
from insar_query import query_data
from gis_viewer import create_map

# Example usage
results = detect_changes(
    data_path="/data",
    algorithm="moderate",
    coherence_threshold=0.7
)
```

## Error Messages

Common error messages and their meanings:

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid bounding box format` | Incorrect bbox syntax | Use lon1,lat1,lon2,lat2 |
| `No data files found` | Missing CSV files | Check path and run query first |
| `Insufficient memory` | Not enough RAM | Reduce threads or data size |
| `Network connection failed` | Internet issues | Check connection and retry |
| `CUDA not available` | GPU setup issues | Use CPU version or fix GPU setup |

## Version Information

Check script versions:

```bash
python insar-query.py --version
python hybrid_detector.py --version
python gis_map_viewer.py --version
```
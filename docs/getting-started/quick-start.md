# Quick Start Guide

Get up and running with InSAR change detection in just a few minutes! This guide will walk you through a complete analysis workflow.

## Overview

The analysis process consists of three main steps:

1. **Query Data** - Download InSAR data for your area
2. **Process Data** - Run change detection algorithms  
3. **Visualize Results** - Explore findings with interactive tools

## Step 1: Query InSAR Data

Start by downloading data for your area of interest using the `insar-query.py` script:

```bash
python insar-query.py --path /data/insar-analysis --bbox=7.986722817,58.149864623,7.991700669,58.146240529
```

### Parameters Explained

- `--path`: Directory where data will be stored
- `--bbox`: Bounding box coordinates (lon1,lat1,lon2,lat2)
- `--period`: Time period (optional, defaults to 2019-2023)


## Step 2: Process the Data

Run change detection analysis on your downloaded data:

```bash title="Process data with moderate sensitivity"
python hybrid_detector.py \
    --path /data/insar-analysis \
    --moderate \
    --cpu-threads 4
```

### Detection Algorithms

Choose one algorithm based on your needs:

| Algorithm | Use Case | Sensitivity |
|-----------|----------|-------------|
| `--ultra-selective` | Only dramatic changes | Very Low |
| `--moderate` | Balanced detection | Medium |
| `--gradual` | Subtle changes | High |
| `--maximum` | All possible changes | Very High |

!!! tip "Recommended Starting Point"
    Use `--moderate` for your first analysis - it provides a good balance between sensitivity and reliability.

## Step 3: Visualize Results

### Interactive Map Viewer

Launch the web-based map viewer to explore your results:

```bash
python gis_map_viewer.py
```

This opens an interactive map in your browser where you can:

- Click on detected change points for details
- Toggle different data layers
- Explore temporal patterns

### Time-Series Plots

Generate detailed plots for specific points:

```bash
python insar-visualizer.py --results change_detection_results.csv --num-points 5 --selection mixed
```

## Expected Output

After processing, you'll have:

- `change_detection_results.csv` - All detected changes with metrics
- `file_mapping.json` - Maps internal IDs to original filenames  
- `visualizations/` - Time-series plots (if generated)

## What's Next?

- Explore the [User Guide](../user-guide/querying-data.md) for detailed parameter explanations
- Learn about [Advanced Features](../advanced/gpu-processing.md) for larger datasets
- Review the [API Reference](../advanced/api-reference.md) for all available options

!!! success "Congratulations!"
    You've completed your first InSAR change detection analysis! The interactive map viewer is your main tool for exploring and understanding the results.
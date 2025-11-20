# InSAR Norway Change Detection

A Python-based toolkit for querying, processing, and analyzing InSAR (Interferometric Synthetic Aperture Radar) data from the Norwegian Ground Motion Service.

## What is InSAR?

InSAR (Interferometric Synthetic Aperture Radar) is a powerful remote sensing technique that uses satellite radar images to detect and measure ground movement with millimeter precision. This toolkit specializes in analyzing Sentinel-1 satellite data to identify ground motion changes over time.

## Key Features

- **Data Querying**: Access multiple Sentinel-1 datasets from InSAR Norway
- **Advanced Analysis**: CPU-based change detection with multiple algorithms
- **Interactive Visualization**: GIS map viewer with historical data overlays
- **Time-Series Analysis**: Detailed plotting for specific measurement points
- **Multi-Track Support**: Both ascending and descending satellite tracks
- **Statistical Analysis**: Temporal changes and coherence measurements

## Quick Start

Get started with the InSAR change detection toolkit in just a few steps:

1. **[Installation](getting-started/installation.md)** - Set up your environment
2. **[Quick Start Guide](getting-started/quick-start.md)** - Run your first analysis
3. **[User Guide](user-guide/querying-data.md)** - Detailed workflows

## Workflow Overview

The analysis process follows three main steps:

1. **Query Data** - Download InSAR data for your area of interest
2. **Process Data** - Run change detection algorithms  
3. **Visualize Results** - Explore results with interactive maps and plots

**[View Detailed Workflow](workflow.md)** - Interactive diagrams and step-by-step process

## Getting Help

- Explore the [Advanced](advanced/gpu-processing.md) section for GPU processing
- Check the [API Reference](advanced/api-reference.md) for detailed command options
- Review the [User Guide](user-guide/querying-data.md) for comprehensive workflows

## About

This project uses data from the Norwegian Ground Motion Service (insar.ngu.no) and is built with open-source libraries including NumPy, Pandas, Flask, GeoPandas, Rasterio, and Matplotlib.
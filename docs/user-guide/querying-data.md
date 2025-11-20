# Querying Data

The `insar-query.py` script is your gateway to accessing InSAR data from the Norwegian Ground Motion Service. This guide covers all available options and best practices.

## Basic Usage

```bash
python insar-query.py --path /path/to/output --bbox=lon1,lat1,lon2,lat2
```

## Command-Line Parameters

### Required Parameters

#### `--path`
Directory where downloaded data will be stored.

```bash
--path /data/insar-analysis
```

#### `--bbox`
Bounding box coordinates defining your area of interest.

**Format**: `longitude1,latitude1,longitude2,latitude2` (WGS84/EPSG:4326)

```bash
--bbox=7.986722817,58.149864623,7.991700669,58.146240529
```

!!! info "Coordinate System Information"
    - **Input**: Decimal degrees in WGS84 (longitude,latitude)
    - **Output**: Both coordinate systems in CSV:
        - `easting`, `northing` in EPSG:3035 (ETRS89-LAEA) - primary
        - `longitude`, `latitude` in WGS84 (EPSG:4326) - converted
    - **EPSG:3035** is used by GIS viewer for higher accuracy in European analysis
    - Keep areas reasonably small for faster processing

### Optional Parameters

#### `--period`
Time period for Sentinel-1 datasets.

**Available options**:
- `2019-2023` (default)
- `2020-2024` 
- `latest`
- `all`

```bash
--period 2020-2024
```

#### `--list-datasets`
Display all available datasets on the server without downloading.

```bash
python insar-query.py --list-datasets
```

## Example

### Small Urban Area
```bash
python insar-query.py \
    --path /data/oslo-center \
    --bbox=10.7300,59.9100,10.7400,59.9200 \
    --period all
```

## Understanding the Data

### File Structure

After querying, your output directory will contain:

```
/path/to/output/
├── dataset1_ascending.csv
├── dataset1_descending.csv
├── dataset2_ascending.csv
└── dataset2_descending.csv
```

### CSV File Contents

Each CSV file contains InSAR measurement points with columns:

| Column | Description |
|--------|-------------|
| `geometry` | Point coordinates (WKT format) |
| `velocity` | Ground motion velocity (mm/year) |
| `coherence` | Data quality measure (0-1) |
| `date` | Measurement date |
| `track` | Satellite track (ascending/descending) |

## Data Quality Considerations

### Coherence Values
- **High (0.8-1.0)**: Excellent data quality
- **Medium (0.6-0.8)**: Good data quality  
- **Low (0.4-0.6)**: Moderate quality, use with caution
- **Very Low (<0.4)**: Poor quality, consider filtering

### Spatial Coverage
- Urban areas: Better coverage due to stable reflectors
- Rural areas: Sparser coverage, especially in vegetation
- Coastal areas: Variable quality depending on conditions

## Troubleshooting

### Common Issues
#### No data returned
- Check bounding box coordinates are valid
- Verify the area has InSAR coverage on [https://insar.ngu.no/](https://insar.ngu.no/)
- Try a different time period

#### Download failures
- Check internet connection
- Verify server availability with `--list-datasets`
- Try smaller bounding box areas

#### Slow downloads
- Large areas take longer to process
- Consider splitting into smaller regions
- Use `latest` period for faster queries

## Best Practices

1. **Start Small**: Begin with small areas to understand data patterns
2. **Check Coverage**: Use `--list-datasets` to verify data availability  
3. **Multiple Periods**: Compare different time periods for comprehensive analysis
4. **Document Queries**: Keep track of bounding boxes and periods used

## Next Steps

Once you have downloaded data, proceed to [Processing Data](processing-data.md) to analyze the InSAR measurements for significant changes.
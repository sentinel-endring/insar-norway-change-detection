# Processing Data

The `hybrid_detector.py` script analyzes your downloaded InSAR data to identify significant ground motion changes. This guide explains all detection algorithms and processing options.

## Basic Usage

```bash
python hybrid_detector.py --path /path/to/downloaded_data --moderate
```

## Detection Algorithms

Choose one algorithm based on your analysis requirements:

### Ultra-Selective (`--ultra-selective`)
**Default algorithm** - Identifies only the most dramatic changes.

```bash title="Ultra-selective detection for high confidence results"
python hybrid_detector.py \
    --path /data/insar \
    --ultra-selective \
    --cpu-threads 8
```

**Best for**:
- Critical infrastructure monitoring
- High-confidence change detection
- Reducing false positives

### Moderate (`--moderate`)
Balanced approach providing reliable results with reasonable sensitivity.

```bash title="Moderate detection - recommended starting point"
python hybrid_detector.py \
    --path /data/insar \
    --moderate \
    --cpu-threads 4
```

**Best for**:
- General-purpose analysis
- First-time users
- Balanced sensitivity vs. reliability

### Gradual (`--gradual`)
Detects subtle and gradual changes over time.

```bash
python hybrid_detector.py --path /data/insar --gradual
```

**Best for**:
- Long-term subsidence monitoring
- Slow environmental changes
- Research applications

### Maximum (`--maximum`)
Extremely sensitive - detects all possible changes.

```bash
python hybrid_detector.py --path /data/insar --maximum
```

**Best for**:
- Comprehensive surveys
- Research requiring all potential signals
- Areas with expected subtle changes

### Baseline (`--baseline`)
Relaxed criteria for baseline detection studies.

```bash
python hybrid_detector.py --path /data/insar --baseline
```

### Original (`--original`)
Legacy algorithm for backward compatibility.

```bash
python hybrid_detector.py --path /data/insar --original
```

## Processing Parameters

### Performance Options

#### `--cpu-threads`
Number of CPU threads for parallel processing.

```bash
--cpu-threads 8
```

!!! tip "Thread Selection"
    Use 50-75% of your available CPU cores for optimal performance without system overload.

#### `--coherence`
Minimum coherence threshold for data quality filtering.

```bash
--coherence 0.7
```

**Coherence Guidelines**:
- `0.8`: Very strict (high quality only)
- `0.7`: Standard (recommended)
- `0.6`: Relaxed (includes more data)
- `0.5`: Permissive (may include noisy data)

### Temporal Options

#### `--fixed-2022-cutoff`
Use fixed date cutoff (2022-01-01) instead of flexible period detection.

```bash
python hybrid_detector.py --path /data/insar --moderate --fixed-2022-cutoff
```

**When to use**:
- Specific event analysis (pre/post 2022)
- Comparative studies
- Known temporal boundaries

## Algorithm Comparison

| Algorithm | Sensitivity | False Positives | Processing Time | Use Case |
|-----------|-------------|-----------------|-----------------|----------|
| Ultra-selective | Very Low | Minimal | Fast | Critical monitoring |
| Moderate | Medium | Low | Medium | General analysis |
| Gradual | High | Medium | Medium | Subtle changes |
| Maximum | Very High | High | Slow | Comprehensive surveys |
| Baseline | Medium-High | Medium | Medium | Research baseline |

## Output Files

Processing generates two main files:

### `change_detection_results.csv`
Contains all detected changes with detailed metrics:

| Column | Description |
|--------|-------------|
| `point_id` | Unique identifier for each point |
| `longitude` | Point longitude |
| `latitude` | Point latitude |
| `change_magnitude` | Magnitude of detected change (mm) |
| `change_date` | Estimated change occurrence |
| `coherence_avg` | Average coherence value |
| `confidence` | Detection confidence score |
| `velocity_before` | Velocity before change (mm/year) |
| `velocity_after` | Velocity after change (mm/year) |

### `file_mapping.json`
Maps internal point IDs to original data filenames for traceability.

## Examples

### Standard Urban Analysis
```bash
python hybrid_detector.py \
    --path /data/oslo-analysis \
    --moderate \
    --cpu-threads 4 \
    --coherence 0.7
```

### High-Sensitivity Environmental Study
```bash
python hybrid_detector.py \
    --path /data/environmental-study \
    --gradual \
    --cpu-threads 8 \
    --coherence 0.6
```

### Infrastructure Monitoring
```bash
python hybrid_detector.py \
    --path /data/infrastructure \
    --ultra-selective \
    --coherence 0.8 \
    --fixed-2022-cutoff
```

## Interpreting Results

### Change Magnitude
- **< 5mm**: Small changes, possibly noise
- **5-15mm**: Moderate changes, investigate further
- **15-30mm**: Significant changes, likely real
- **> 30mm**: Large changes, high confidence

### Confidence Scores
- **> 0.8**: High confidence detection
- **0.6-0.8**: Medium confidence
- **0.4-0.6**: Low confidence, verify manually
- **< 0.4**: Very uncertain, possibly false positive

## Performance Optimization

### Large Datasets
- Use more CPU threads (`--cpu-threads`)
- Start with ultra-selective algorithm
- Process in smaller spatial chunks
- Increase coherence threshold for speed

### Memory Considerations
- Large areas may require significant RAM
- Monitor system resources during processing
- Consider processing subregions separately

## Troubleshooting

### Common Issues

**No changes detected**
- Try a more sensitive algorithm (`--gradual` or `--maximum`)
- Lower coherence threshold (`--coherence 0.6`)
- Check data quality and coverage

**Too many false positives**
- Use stricter algorithm (`--ultra-selective`)
- Increase coherence threshold (`--coherence 0.8`)
- Check data quality in problematic areas

**Slow processing**
- Reduce CPU threads if system becomes unresponsive
- Process smaller areas
- Use ultra-selective algorithm for faster results

**Memory errors**
- Process smaller spatial regions
- Increase system swap space
- Use fewer CPU threads

## Next Steps

After processing, proceed to [Visualization](visualization.md) to explore your detected changes using interactive maps and time-series plots.
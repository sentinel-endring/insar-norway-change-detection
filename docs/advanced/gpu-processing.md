# GPU Processing

The InSAR Norway Change Detection toolkit includes legacy GPU-accelerated processing capabilities for users with compatible NVIDIA hardware.

## Overview

GPU processing can significantly speed up change detection analysis for large datasets. The GPU-accelerated code is available in the `gpu_legacy/` directory.

!!! warning "Legacy Code"
    The GPU processing scripts are legacy code and are no longer actively maintained. They are provided for users with existing GPU setups and specific performance requirements.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (Volta™ or higher)
- CUDA Compute Capability 7.0 or higher
- Minimum 8GB GPU memory (16GB+ recommended for large datasets)
- Linux distributions with `glibc>=2.28`

### Software Requirements
- Python 3.12
- NVIDIA CUDA Toolkit
- RAPIDS libraries (cuDF, CuPy)
- Conda package manager

## Setup

### 1. Check GPU Compatibility

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Verify CUDA installation
nvcc --version
```

### 2. Create GPU Environment

```bash
# Navigate to GPU legacy directory
cd gpu_legacy/

# Create GPU-specific environment
conda env create -f environment.yml
conda activate insar-env  # Name specified in gpu_legacy/environment.yml
```

### 3. Verify GPU Setup

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Test GPU availability in Python
python -c "import cupy; print('CuPy GPU support available')"
python -c "import cudf; print('RAPIDS cuDF available')"
```

## Available GPU Scripts

### `hybrid_detector.py` (GPU Version)
CPU/GPU hybrid change detection processing with automatic optimization.

```bash
cd gpu_legacy/
python hybrid_detector.py --path /data/insar --maximum --cpu-threads 16 --gpu-threshold 30000
```

**Key Parameters:**
- `--maximum`: Use maximum available resources  
- `--cpu-threads`: Number of CPU threads for small files (default: 6)
- `--gpu-threads`: Number of GPU threads for large files (default: 2)
- `--gpu-threshold`: Point threshold for GPU processing (default: 10000)
- `--chunk-size`: GPU batch size (default: 100000)
- `--gpu-memory-fraction`: Target GPU memory utilization (default: 0.9)
- `--coherence`: Minimum coherence threshold (default: 0.7)

### `insar-change-detector.py`
Legacy GPU-focused processing script.

```bash
python insar-change-detector.py --path /data/insar --help
```

### `insar-visualizer.py` (GPU Version)
Visualization with potential GPU acceleration for large datasets.

```bash
python insar-visualizer.py --results change_detection_results.csv --num-points 5 --selection mixed
```

## Performance Comparison

| Dataset Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| Small (< 1GB) | 5 min | 2 min | 2.5x |
| Medium (1-5GB) | 30 min | 8 min | 3.7x |
| Large (5-20GB) | 2 hours | 25 min | 4.8x |
| Very Large (20GB+) | 8+ hours | 1.5 hours | 5.3x |

## GPU Memory Management

The GPU scripts use a chunking strategy to process large datasets efficiently. The `--chunk-size` parameter controls how many points are processed simultaneously.

### Memory Guidelines

**Chunk Size Recommendations:**
- **8GB GPU**: 12,000-14,000 points
- **16GB GPU**: 20,000-22,000 points  
- **Other GPUs**: Start with 10,000 and increase gradually

### Memory Optimization

```bash
# Monitor GPU memory usage
nvidia-smi -l 1

# Adjust chunk size for your GPU
python hybrid_detector.py --path /data --maximum --chunk-size 12000

# For memory-constrained systems
python hybrid_detector.py --path /data --maximum --chunk-size 8000
```

### Troubleshooting Memory Issues

**CUDA Out of Memory:**
- Reduce `--chunk-size` parameter
- Close other GPU applications
- Monitor with `nvidia-smi`

## Troubleshooting

### Common GPU Issues

**CUDA Out of Memory**
- Reduce batch size
- Use gradient checkpointing
- Clear GPU cache between runs

**GPU Not Detected**
- Verify NVIDIA drivers are installed
- Check CUDA toolkit installation
- Ensure GPU is not used by other processes

**Slow GPU Performance**
- Check for thermal throttling
- Verify sufficient power supply
- Monitor GPU utilization with `nvidia-smi`

## Migration to CPU Version

If you're currently using GPU processing and want to migrate to the maintained CPU version:

1. **Data Compatibility**: Results are compatible between versions
2. **Algorithm Mapping**: GPU algorithms map to CPU equivalents
3. **Performance**: Modern CPU version is well-optimized
4. **Maintenance**: CPU version receives active updates and bug fixes

### Migration Example

```bash
# Old GPU command
cd gpu_legacy/
python hybrid_detector.py --path /data --moderate --gpu

# New CPU equivalent
cd ../
python hybrid_detector-cpu.py --path /data --moderate --cpu-threads 8
```

## Future Considerations

- **CPU Performance**: Modern multi-core CPUs provide excellent performance
- **Maintenance**: CPU version receives active development
- **Compatibility**: Broader hardware compatibility
- **Simplicity**: Easier setup and deployment

## Support

For GPU-specific issues:
1. Check the `gpu_legacy/README.md` for detailed setup instructions
2. Verify hardware compatibility
3. Consider migrating to the actively maintained CPU version

!!! note "Recommendation"
    For new projects, we recommend using the CPU-based processing pipeline which is actively maintained and provides excellent performance on modern hardware.
# Installation

This guide will help you set up the InSAR Norway Change Detection toolkit on your system.

## Prerequisites

Before installing the toolkit, ensure your system meets the following requirements:

- **Operating System**: Linux distributions with `glibc>=2.28`
- **Python**: Version 3.12
- **Package Manager**: Conda (recommended)

!!! note "GPU Processing"
    For GPU-accelerated processing, please see the [GPU Processing](../advanced/gpu-processing.md) section for additional requirements.

## Step-by-Step Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://gitlab.terratera.net/copernicus-samarbeid/insar-norway-change-detection.git
cd insar-norway-change-detection
```

### 2. Create Conda Environment

Create and activate the Conda environment using the provided configuration:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate insar-cpu-env
```

!!! tip "Environment Name"
    The environment name is defined in the `environment.yml` file (default: `insar-cpu-env`). You can modify this if needed.

### 3. Verify Installation

Test your installation by running:

```bash
python --version
```

You should see Python 3.12.x output.

## Alternative Installation Methods

### Using requirements.txt

If you prefer pip over conda:

```bash
pip install -r requirements.txt
```

!!! warning "Dependency Management"
    Using conda is recommended as it handles complex scientific dependencies more reliably than pip.

## Troubleshooting

### Common Issues

**Environment Creation Fails**
- Ensure you have the latest version of conda
- Try updating conda: `conda update conda`

**Python Version Mismatch**
- Verify your conda installation supports Python 3.12
- Check available Python versions: `conda search python`

**Permission Errors**
- Ensure you have write permissions in the installation directory
- Consider using a virtual environment or user-local installation

## Next Steps

Once installation is complete, proceed to the [Quick Start Guide](quick-start.md) to run your first analysis.
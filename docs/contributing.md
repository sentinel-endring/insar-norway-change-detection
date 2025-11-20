# Contributing to InSAR Norway Change Detection

Thank you for your interest in contributing to the InSAR Norway Change Detection project! This guide outlines how you can help improve the toolkit.

## Ways to Contribute

### Bug Reports
Found an issue? Help us improve by reporting bugs:

1. **Check existing issues** to avoid duplicates
2. **Use the issue template** when creating new reports
3. **Include system information**: OS, Python version, environment details
4. **Provide reproducible examples** when possible

### Documentation
Help make the toolkit more accessible:

- **Fix typos or unclear instructions**
- **Add examples for common use cases**
- **Improve command explanations**
- **Translate content** (when applicable)

### Code Contributions

#### Getting Started
```bash title="Set up development environment"
# Clone the repository
git clone https://github.com/sentinel-endring/insar-norway-change-detection.git
cd insar-norway-change-detection

# Create development environment
conda env create -f environment-cpu.yml
conda activate insar-cpu-env

# Install additional development dependencies
pip install mkdocs-material[imaging]
```

#### Development Workflow
1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the coding standards
4. **Test your changes** using the existing test suite
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

#### Coding Standards
- **Follow PEP 8** for Python code style
- **Add docstrings** for new functions and classes
- **Include type hints** where appropriate
- **Write tests** for new functionality
- **Keep functions focused** and well-documented

```python title="Example function with proper documentation"
def detect_changes(data: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect significant changes in InSAR time series data.
    
    Args:
        data: DataFrame containing InSAR measurements
        threshold: Statistical threshold for change detection
        
    Returns:
        DataFrame with detected changes and confidence scores
    """
    # Implementation here
    pass
```

### Testing
Help ensure reliability:

- **Run existing tests**: Tests are executed via GitHub Actions
- **Add test cases** for new features
- **Test on different systems** if possible
- **Validate with real data** when contributing algorithms

### Algorithm Improvements
Contribute to the change detection algorithms:

- **Optimize performance** for large datasets  
- **Improve accuracy** of detection methods
- **Add new algorithms** with proper documentation
- **Enhance spatial analysis** capabilities

## Documentation Contributions

### Local Documentation Development
```bash title="Work on documentation locally"
# Start the documentation server
mkdocs serve

# Build the documentation
mkdocs build
```

Visit `http://127.0.0.1:8000` to preview changes.

### Documentation Structure
```
docs/
├── index.md                 # Homepage
├── workflow.md             # Process overview  
├── getting-started/        # Installation & quick start
├── user-guide/            # Detailed usage guides
├── advanced/              # GPU processing & API reference
└── images/                # Screenshots and diagrams
```

## Pull Request Guidelines

### Before Submitting
- [ ] **Test your changes** locally
- [ ] **Update documentation** if needed
- [ ] **Follow coding standards**
- [ ] **Write clear commit messages**

### Pull Request Template
```markdown title="PR Description Template"
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] Documentation builds correctly
- [ ] Changes tested with real data (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Request Reviews**: Code-specific feedback

### Development Resources
- **MkDocs**: Documentation framework ([mkdocs.org](https://www.mkdocs.org))
- **Material Theme**: UI components ([squidfunk.github.io](https://squidfunk.github.io/mkdocs-material/))
- **InSAR Norway API**: Data source ([insar.ngu.no](https://insar.ngu.no))

## Recognition

All contributors are recognized in our project documentation. Significant contributions may be highlighted in release notes.

---

**Ready to contribute?** Check out our [open issues](https://github.com/sentinel-endring/insar-norway-change-detection/issues) for good first contributions!
# Installation

## Prerequisites

VortexNN requires:

- Python 3.7 or later
- PyTorch 1.7.0 or later
- NumPy 1.19.0 or later
- scikit-learn 0.23.0 or later
- matplotlib 3.3.0 or later (for visualization)

## Installing from PyPI

VortexNN is not yet available on PyPI. When it becomes available, you'll be able to install it with:

```bash
pip install vortexnn
```

## Installing from Source

To install the development version from source:

```bash
# Clone the repository
git clone https://github.com/vortexnn/vortexnn.git
cd vortexnn

# Install in development mode
pip install -e .
```

This will install VortexNN in editable mode, allowing you to modify the source code and have the changes reflected immediately.

## Verifying Installation

To verify that VortexNN is installed correctly, you can run:

```python
import vortexnn
print(vortexnn.__version__)
```

## Installing for Development

If you plan to contribute to VortexNN, you should install the development dependencies:

```bash
pip install -e ".[dev]"
```

Or install the test dependencies separately:

```bash
pip install pytest pytest-cov
```

You can then run the tests with:

```bash
pytest
```

## Troubleshooting

### CUDA Compatibility

VortexNN relies on PyTorch for GPU acceleration. If you encounter CUDA-related issues, make sure your PyTorch installation is compatible with your CUDA version.

### Missing Dependencies

If you encounter errors about missing dependencies, try installing the requirements explicitly:

```bash
pip install -r requirements.txt
```

### Other Issues

If you encounter other installation issues, please check the [GitHub issues](https://github.com/vortexnn/vortexnn/issues) or create a new issue with details about your environment and the error you're experiencing.

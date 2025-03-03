# VortexNN Documentation

Welcome to the VortexNN documentation. VortexNN is a PyTorch-based library implementing specialized neural network architectures inspired by vortex mathematics principles.

## Contents

- [Installation](installation.md)
- [Getting Started](getting_started.md)
- [API Reference](api/index.md)
- [Advanced Usage](advanced_usage.md)
- [Mathematical Background](mathematical_background.md)
- [Contributing](contributing.md)

## Quick Start

```python
import torch
from vortexnn import AdvancedVortexGNN, generate_cyclical_data, train_vortex_model

# Generate data
X, y = generate_cyclical_data(n_samples=1000, input_dim=16)

# Create model
model = AdvancedVortexGNN(input_dim=16, hidden_dim=64, output_dim=1)

# Train model (simplified example)
# See 'Getting Started' for complete example
```

## About Vortex Mathematics

Vortex mathematics is a framework that explores numerical patterns emerging from digital roots and modular arithmetic. Key concepts include:

- **Doubling Cycle**: The sequence 1→2→4→8→7→5→1
- **Complementary Pairs**: Number pairs that sum to 9 (1-8, 2-7, 4-5, 3-6)
- **Digital Roots**: Single-digit values obtained by repeatedly summing digits

These mathematical principles provide unique structural patterns that can be encoded into neural networks to enhance their pattern recognition capabilities.

## License

VortexNN is released under the MIT License.

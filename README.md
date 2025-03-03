# VortexNN

## Vortex-inspired Graph Neural Networks

VortexNN is a PyTorch-based library that implements specialized neural network architectures inspired by vortex mathematics principles. The library provides a collection of unique layers, models, and utilities for pattern recognition and time series analysis.

## Key Features

- **Specialized Layers**: Various implementations of vortex-based neural network layers
  - Hyperbolic Geometry Layers
  - Quaternion-based Layers
  - Wavelet-based Layers
  - Modular Field Layers
  - Vortex Attention Mechanisms

- **Advanced Models**: Complete model implementations
  - Advanced Vortex GNN (Graph Neural Network)

- **Utility Functions**:
  - Specialized training procedures
  - Evaluation metrics
  - Synthetic data generators with vortex patterns
  - Vortex-inspired learning rate schedulers
  - Specialized regularization techniques

## Installation

```bash
# Install from PyPI (not yet available)
# pip install vortexnn

# Install from source
git clone https://github.com/drazzam/vortexnn.git
cd vortexnn
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from vortexnn import (
    AdvancedVortexGNN,
    train_vortex_model,
    evaluate_vortex_model,
    generate_cyclical_data
)

# Generate synthetic data with vortex patterns
X, y = generate_cyclical_data(n_samples=1000, input_dim=16)

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Create data loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))
test_loader = DataLoader(test_dataset, batch_size=32)

# Create model
model = AdvancedVortexGNN(
    input_dim=X_train.shape[1],
    hidden_dim=64,
    output_dim=1,
    num_layers=3
)

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

history = train_vortex_model(
    model=model,
    train_loader=train_loader,
    valid_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=100,
    patience=15
)

# Evaluate model
metrics, y_pred, y_true = evaluate_vortex_model(
    model=model,
    X_test=X_test_scaled,
    y_test=y_test_scaled,
    scaler_y=scaler_y,
    device=device
)

print(metrics)
```

## Advanced Usage

See the `examples/` directory for more detailed usage examples.

## Mathematical Background

VortexNN builds upon principles from vortex mathematics, which studies numerical patterns and symmetries related to digital roots and modular arithmetic. Key concepts include:

- **Doubling Cycle**: The sequence 1→2→4→8→7→5→1 that emerges when repeatedly doubling a number and taking the digital root
- **Complementary Pairs**: Number pairs that sum to 9 (1-8, 2-7, 4-5, 3-6)
- **Digital Roots**: The single-digit value obtained by repeatedly summing the digits of a number

These mathematical principles are encoded into the neural network architecture to enhance pattern recognition capabilities.

## License

MIT License

Copyright (c) 2025 Ahmed Y. Azzam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

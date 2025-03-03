# Getting Started with VortexNN

This guide will walk you through the basic usage of VortexNN, from data preparation to model training and evaluation.

## Basic Example

Here's a complete example showing how to train and evaluate a Vortex Neural Network:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Import VortexNN components
from vortexnn import (
    AdvancedVortexGNN,
    train_vortex_model,
    evaluate_vortex_model,
    generate_cyclical_data,
    VortexCyclicalLR
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic data with vortex patterns
X, y = generate_cyclical_data(n_samples=1000, input_dim=16)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the Vortex GNN model
model = AdvancedVortexGNN(
    input_dim=X_train.shape[1],
    hidden_dim=64,
    output_dim=1,
    num_layers=3
)

# Set up loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Create learning rate scheduler with vortex pattern
scheduler = VortexCyclicalLR(
    optimizer,
    base_lr=0.0001,
    max_lr=0.001,
    step_size_up=500,
    cycle_pattern=[1, 2, 4, 8, 7, 5]
)

# Train the model
history = train_vortex_model(
    model=model,
    train_loader=train_loader,
    valid_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=50,
    patience=10,
    scheduler=scheduler,
    use_regularization=True
)

# Evaluate the model
metrics, y_pred, y_true = evaluate_vortex_model(
    model=model,
    X_test=X_test_scaled,
    y_test=y_test_scaled,
    scaler_y=scaler_y,
    device=device
)

# Print metrics
print("\nModel Performance Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.6f}")

# Visualize results
plt.figure(figsize=(12, 5))

# Plot training history
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot predictions vs true values
plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')

plt.tight_layout()
plt.show()
```

## Using Real Data

To use VortexNN with your own data:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load your data
data = pd.read_csv('your_data.csv')

# Prepare features and target
X = data.drop('target_column', axis=1).values
y = data['target_column'].values.reshape(-1, 1)

# Split, scale, and prepare data as shown in the basic example

# Create and train model
model = AdvancedVortexGNN(
    input_dim=X.shape[1],
    hidden_dim=64,
    output_dim=1
)

# Train and evaluate as shown in the basic example
```

## Next Steps

- Explore the [Advanced Usage](advanced_usage.md) guide for more complex scenarios
- Check the [API Reference](api/index.md) for detailed documentation of all components
- Learn about the [Mathematical Background](mathematical_background.md) of VortexNN

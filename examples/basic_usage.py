"""
Basic example of using the VortexNN library.

This script demonstrates how to:
1. Create synthetic data with vortex patterns
2. Train an Advanced Vortex GNN model
3. Evaluate the model's performance
4. Visualize the results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Import from VortexNN library
from vortexnn import (
    AdvancedVortexGNN,
    train_vortex_model,
    evaluate_vortex_model,
    generate_cyclical_data,
    VortexCyclicalLR
)


def main():
    """Main function demonstrating VortexNN usage."""
    print("VortexNN Basic Usage Example")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y = generate_cyclical_data(n_samples=2000, input_dim=16)
    
    # Split data
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
    print("\nInitializing Advanced Vortex GNN...")
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
    print("\nTraining model...")
    max_epochs = 50  # Reduced for the example
    history = train_vortex_model(
        model=model,
        train_loader=train_loader,
        valid_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=max_epochs,
        patience=15,
        scheduler=scheduler,
        use_regularization=True
    )
    
    # Evaluate the model
    print("\nEvaluating model performance...")
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
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Training loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Predictions vs true values
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    
    # Training time per epoch
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch_times'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    
    # Sample predictions over time
    plt.subplot(2, 2, 4)
    sample_size = min(100, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    indices.sort()
    plt.plot(y_true[indices], 'b-', label='True Values')
    plt.plot(y_pred[indices], 'r--', label='Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Sample Predictions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vortex_gnn_results.png')
    plt.close()
    
    print("\nResults visualization saved as 'vortex_gnn_results.png'")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

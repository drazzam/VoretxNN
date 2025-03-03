"""
Training utilities for Vortex Neural Networks.
"""

import torch
import time


def train_vortex_model(model, train_loader, valid_loader, criterion, optimizer, device,
                     epochs=100, patience=15, scheduler=None, use_regularization=True):
    """
    Train a Vortex Neural Network model with specialized training procedures.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Maximum number of epochs
        patience: Early stopping patience
        scheduler: Optional learning rate scheduler
        use_regularization: Whether to use vortex-specific regularization

    Returns:
        history: Dictionary containing training metrics
    """
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_times": []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training {model.__class__.__name__}...")

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        reg_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            # Compute loss
            loss = criterion(outputs, y_batch)

            # Add regularization if supported by the model
            if use_regularization and hasattr(model, 'compute_regularization'):
                # Get the node features (implementation may vary based on model architecture)
                if hasattr(model, 'node_features'):
                    node_features = model.node_features
                    r_loss = model.compute_regularization(node_features, y_batch)
                    loss += r_loss
                    reg_loss += r_loss.item()

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in batch. Skipping.")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Track loss
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_reg_loss = reg_loss / len(train_loader) if reg_loss > 0 else 0

        history["train_loss"].append(avg_train_loss)

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward pass
                outputs = model(X_batch)

                # Compute loss
                loss = criterion(outputs, y_batch)

                # Track loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        history["val_loss"].append(avg_val_loss)

        # Track time
        epoch_time = time.time() - start_time
        history["epoch_times"].append(epoch_time)

        # Print progress
        reg_info = f", Reg Loss: {avg_reg_loss:.6f}" if avg_reg_loss > 0 else ""
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f}{reg_info} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    return history

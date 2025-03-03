"""
Evaluation utilities for Vortex Neural Networks.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_vortex_model(model, X_test, y_test, scaler_y=None, device='cpu'):
    """
    Evaluate a Vortex Neural Network model.

    Args:
        model: The trained model
        X_test: Test features
        y_test: Test targets
        scaler_y: Optional scaler for the target variable
        device: Device to evaluate on

    Returns:
        metrics: Dictionary of evaluation metrics
        y_pred: Model predictions
        y_test: Actual test values
    """
    model.eval()
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    try:
        with torch.no_grad():
            y_pred = model(X_test)

        # Check for NaN values in predictions
        if torch.isnan(y_pred).any():
            print("Warning: NaN values in predictions. Replacing with zeros.")
            y_pred = torch.nan_to_num(y_pred, nan=0.0)

        # Move to CPU for evaluation
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Inverse transform if scaler provided
        if scaler_y is not None:
            y_pred_np = scaler_y.inverse_transform(y_pred_np)
            y_test_np = scaler_y.inverse_transform(y_test_np)

        # Calculate metrics
        mse = mean_squared_error(y_test_np, y_pred_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        r2 = r2_score(y_test_np, y_pred_np)

        metrics = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        }

        return metrics, y_pred_np, y_test_np

    except Exception as e:
        print(f"Error during evaluation: {e}")
        metrics = {
            "MSE": float('nan'),
            "RMSE": float('nan'),
            "MAE": float('nan'),
            "R²": float('nan')
        }
        return metrics, None, y_test.cpu().numpy()

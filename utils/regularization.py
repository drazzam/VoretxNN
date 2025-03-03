"""
Specialized regularization functions for Vortex Neural Networks.
"""

import torch
import torch.nn.functional as F


def complementary_regularization_loss(node_features, lambda_reg=0.01):
    """
    Regularization that encourages complementary nodes to develop related representations.
    
    Args:
        node_features: Node features of shape [batch_size, num_nodes, hidden_dim]
        lambda_reg: Regularization strength
        
    Returns:
        Regularization loss
    """
    loss = 0.0
    # Define complementary pairs (1-8, 2-7, etc.)
    pairs = [(0,7), (1,6), (3,4), (2,5)]
    for a, b in pairs:
        if a < node_features.size(1) and b < node_features.size(1):
            # Encourage symmetry in activations
            diff = node_features[:, a] - (1 - node_features[:, b])
            loss += torch.mean(diff**2)
    return lambda_reg * loss


def digital_root_consistency_loss(predictions, targets, lambda_dr=0.01):
    """
    Regularization that encourages predictions to maintain digital root consistency.
    
    Args:
        predictions: Model predictions
        targets: Target values
        lambda_dr: Regularization strength
        
    Returns:
        Digital root consistency loss
    """
    # Calculate digital roots (simplified)
    pred_sum = torch.sum(torch.abs(predictions), dim=-1)
    target_sum = torch.sum(torch.abs(targets), dim=-1)

    pred_dr = torch.fmod(pred_sum, 9.0)
    target_dr = torch.fmod(target_sum, 9.0)

    # Handle zero case (digital root of 0 is 9)
    pred_dr = torch.where(pred_dr == 0, torch.tensor(9.0, device=predictions.device), pred_dr)
    target_dr = torch.where(target_dr == 0, torch.tensor(9.0, device=targets.device), target_dr)

    # Consistency loss
    dr_loss = F.mse_loss(pred_dr, target_dr)

    return lambda_dr * dr_loss

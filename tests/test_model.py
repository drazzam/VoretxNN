"""
Tests for vortex neural network models.
"""

import pytest
import torch
import numpy as np

from vortexnn.models import AdvancedVortexGNN


@pytest.fixture
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


def test_advanced_vortex_gnn_creation(set_seed):
    """Test the creation of AdvancedVortexGNN."""
    input_dim = 16
    hidden_dim = 20
    output_dim = 1
    num_layers = 2
    
    # Create model
    model = AdvancedVortexGNN(input_dim, hidden_dim, output_dim, num_layers)
    
    # Check model properties
    assert model.input_dim == input_dim
    assert model.hidden_dim % 4 == 0, "Hidden dimension should be adjusted to be divisible by 4"
    assert model.output_dim == output_dim
    assert model.num_layers == num_layers
    
    # Check layers creation
    assert len(model.vortex_layers) == num_layers
    assert len(model.attention_layers) == num_layers
    assert len(model.layer_norms) == num_layers


def test_advanced_vortex_gnn_forward(set_seed):
    """Test the forward pass of AdvancedVortexGNN."""
    batch_size = 4
    input_dim = 16
    hidden_dim = 20
    output_dim = 1
    num_layers = 2
    
    # Create model
    model = AdvancedVortexGNN(input_dim, hidden_dim, output_dim, num_layers)
    
    # Create input tensor
    inputs = torch.randn(batch_size, input_dim)
    
    # Forward pass
    outputs = model(inputs)
    
    # Check output shape
    assert outputs.shape == (batch_size, output_dim)
    
    # Check no NaNs in output
    assert not torch.isnan(outputs).any()
    
    # Check node_features storage for regularization
    assert model.node_features is not None
    assert model.node_features.shape == (batch_size, 9, model.hidden_dim)


def test_advanced_vortex_gnn_regularization(set_seed):
    """Test the regularization functionality of AdvancedVortexGNN."""
    batch_size = 4
    input_dim = 16
    hidden_dim = 20
    output_dim = 1
    
    # Create model
    model = AdvancedVortexGNN(input_dim, hidden_dim, output_dim)
    
    # Create input and target tensors
    inputs = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, output_dim)
    
    # Forward pass to populate node_features
    outputs = model(inputs)
    
    # Compute regularization without targets
    reg_loss = model.compute_regularization(model.node_features)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0  # Should be a scalar
    
    # Compute regularization with targets
    reg_loss_with_targets = model.compute_regularization(model.node_features, targets)
    assert isinstance(reg_loss_with_targets, torch.Tensor)
    assert reg_loss_with_targets.ndim == 0  # Should be a scalar

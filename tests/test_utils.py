"""
Tests for vortex neural network utilities.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from vortexnn.models import AdvancedVortexGNN
from vortexnn.utils import (
    generate_cyclical_data,
    generate_complementary_data,
    VortexCyclicalLR,
    complementary_regularization_loss,
    digital_root_consistency_loss
)


@pytest.fixture
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


def test_data_generation(set_seed):
    """Test data generation utilities."""
    n_samples = 100
    input_dim = 9
    
    # Test cyclical data generation
    X_cyclical, y_cyclical = generate_cyclical_data(n_samples, input_dim)
    assert X_cyclical.shape == (n_samples, input_dim)
    assert y_cyclical.shape == (n_samples, 1)
    
    # Test complementary data generation
    X_comp, y_comp = generate_complementary_data(n_samples, input_dim)
    assert X_comp.shape == (n_samples, input_dim)
    assert y_comp.shape == (n_samples, 1)
    
    # Check that the generated data is different
    assert not np.array_equal(X_cyclical, X_comp)
    assert not np.array_equal(y_cyclical, y_comp)


def test_vortex_cyclical_lr(set_seed):
    """Test VortexCyclicalLR scheduler."""
    # Create small model for testing
    model = AdvancedVortexGNN(input_dim=8, hidden_dim=16, output_dim=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create scheduler
    base_lr = 0.0001
    max_lr = 0.001
    scheduler = VortexCyclicalLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=max_lr,
        step_size_up=100,
        cycle_pattern=[1, 2, 4, 8, 7, 5]
    )
    
    # Initial learning rate should be base_lr
    assert optimizer.param_groups[0]['lr'] == base_lr
    
    # Step scheduler multiple times and check that lr changes
    learning_rates = []
    for _ in range(50):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        assert base_lr <= current_lr <= max_lr
    
    # Verify that learning rates are not all the same (scheduler is working)
    assert len(set(learning_rates)) > 1


def test_regularization_functions(set_seed):
    """Test regularization loss functions."""
    batch_size = 4
    num_nodes = 9
    hidden_dim = 8
    output_dim = 1
    
    # Create node features and targets for testing
    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    predictions = torch.randn(batch_size, output_dim)
    targets = torch.randn(batch_size, output_dim)
    
    # Test complementary regularization
    comp_loss = complementary_regularization_loss(node_features)
    assert isinstance(comp_loss, torch.Tensor)
    assert comp_loss.ndim == 0  # Should be a scalar
    assert comp_loss.item() >= 0  # Loss should be non-negative
    
    # Test digital root consistency regularization
    dr_loss = digital_root_consistency_loss(predictions, targets)
    assert isinstance(dr_loss, torch.Tensor)
    assert dr_loss.ndim == 0  # Should be a scalar
    assert dr_loss.item() >= 0  # Loss should be non-negative

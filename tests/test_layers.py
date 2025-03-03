"""
Tests for vortex neural network layers.
"""

import pytest
import torch
import numpy as np

from vortexnn.layers import (
    BaseVortexLayer,
    VortexLayer,
    DualFlowVortexLayer,
    HarmonicResonanceLayer,
    DigitalRootUnit,
    VortexAttention,
    HyperbolicVortexLayer,
    QuaternionVortexLayer,
    WaveletVortexLayer,
    ModularFieldVortexLayer
)


@pytest.fixture
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


def test_base_vortex_layer(set_seed):
    """Test the BaseVortexLayer functionality."""
    batch_size = 4
    num_nodes = 9
    hidden_dim = 8
    
    # Create layer
    layer = BaseVortexLayer(num_nodes, hidden_dim)
    
    # Check adjacency matrix creation
    assert hasattr(layer, 'adjacency')
    assert layer.adjacency.shape == (num_nodes, num_nodes)
    
    # Check weights initialization
    assert len(layer.neighbor_weights) == num_nodes
    assert len(layer.self_weights) == num_nodes
    assert len(layer.bias) == num_nodes
    assert len(layer.layer_norms) == num_nodes


def test_vortex_layer_forward(set_seed):
    """Test the forward pass of VortexLayer."""
    batch_size = 4
    num_nodes = 9
    hidden_dim = 8
    
    # Create layer
    layer = VortexLayer(num_nodes, hidden_dim)
    
    # Create input tensor
    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    
    # Forward pass
    output = layer(node_features)
    
    # Check output shape
    assert output.shape == (batch_size, num_nodes, hidden_dim)
    
    # Check no NaNs in output
    assert not torch.isnan(output).any()


def test_harmonic_resonance_layer(set_seed):
    """Test HarmonicResonanceLayer functionality."""
    batch_size = 4
    num_nodes = 9
    hidden_dim = 8
    
    # Create layer
    layer = HarmonicResonanceLayer(hidden_dim)
    
    # Create input tensor
    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    
    # Forward pass
    output = layer(node_features)
    
    # Check output shape
    assert output.shape == (batch_size, num_nodes, hidden_dim)
    
    # Check with time context
    time_context = torch.randn(batch_size, 1)
    output_with_time = layer(node_features, time_context)
    assert output_with_time.shape == (batch_size, num_nodes, hidden_dim)


def test_digital_root_unit(set_seed):
    """Test DigitalRootUnit functionality."""
    batch_size = 4
    input_dim = 16
    hidden_dim = 8
    
    # Create layer
    layer = DigitalRootUnit(input_dim, hidden_dim)
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = layer(input_tensor)
    
    # Check output shape (should have 9 nodes for vortex structure)
    assert output.shape == (batch_size, 9, hidden_dim)


def test_quaternion_vortex_layer(set_seed):
    """Test QuaternionVortexLayer functionality."""
    batch_size = 4
    num_nodes = 9
    hidden_dim = 8  # Should adjust to be divisible by 4
    
    # Create layer - it should adjust hidden_dim automatically
    layer = QuaternionVortexLayer(num_nodes, hidden_dim)
    
    # Create input tensor
    node_features = torch.randn(batch_size, num_nodes, layer.hidden_dim)
    
    # Forward pass
    output = layer(node_features)
    
    # Check output shape
    assert output.shape == (batch_size, num_nodes, layer.hidden_dim)
    assert layer.hidden_dim % 4 == 0, "Hidden dimension should be divisible by 4"

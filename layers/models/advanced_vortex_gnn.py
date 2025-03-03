"""
Advanced Vortex Graph Neural Network implementation.

This module provides the main implementation of the Advanced Vortex GNN,
which combines multiple specialized layers to create a comprehensive model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vortexnn.layers import (
    HyperbolicVortexLayer,
    QuaternionVortexLayer,
    WaveletVortexLayer,
    ModularFieldVortexLayer,
    VortexAttention,
    HarmonicResonanceLayer,
    DigitalRootUnit,
    DigitalRootNorm
)


class AdvancedVortexGNN(nn.Module):
    """
    Comprehensive Vortex Neural Network that integrates all advanced components.
    This model combines multiple mathematical frameworks to optimize pattern recognition.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(AdvancedVortexGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Ensure hidden_dim is divisible by 4 for quaternion operations
        if hidden_dim % 4 != 0:
            hidden_dim = ((hidden_dim // 4) + 1) * 4
            print(f"Adjusted hidden_dim to {hidden_dim} to be divisible by 4 for quaternion operations")

        self.hidden_dim = hidden_dim

        # Store node features for regularization
        self.node_features = None

        # Input projection using digital root processing
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim, 9 * hidden_dim),
            nn.LeakyReLU(0.01)
        )

        # Digital root processor
        self.digital_root_unit = DigitalRootUnit(9 * hidden_dim, hidden_dim)

        # Specialized layers leveraging different mathematical frameworks
        self.vortex_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: Hyperbolic geometry for cyclical patterns
                self.vortex_layers.append(HyperbolicVortexLayer(9, hidden_dim))
            elif i == 1:
                # Second layer: Quaternion operations for directional flow
                self.vortex_layers.append(QuaternionVortexLayer(9, hidden_dim))
            else:
                # Alternate between wavelet and modular field layers
                if i % 2 == 0:
                    self.vortex_layers.append(WaveletVortexLayer(9, hidden_dim))
                else:
                    self.vortex_layers.append(ModularFieldVortexLayer(9, hidden_dim))

        # Attention mechanism
        self.attention_layers = nn.ModuleList([
            VortexAttention(9, hidden_dim) for _ in range(num_layers)
        ])

        # Harmonic resonance for time-dependent patterns
        self.harmonic_processor = HarmonicResonanceLayer(hidden_dim)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            DigitalRootNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Flow balance parameter (learned)
        self.flow_balance = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, time_context=None):
        """
        Forward pass through the advanced Vortex GNN.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            time_context: Optional time context for harmonic processing

        Returns:
            output: Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # Project input to intermediate space
        projected = self.input_processor(x)

        # Process through digital root unit - already returns shape [batch_size, 9, hidden_dim]
        node_features = self.digital_root_unit(projected)

        # Process through vortex layers with attention and normalization
        for i in range(self.num_layers):
            # Apply vortex layer
            vortex_out = self.vortex_layers[i](node_features)

            # Apply attention mechanism
            attention_out = self.attention_layers[i](vortex_out)

            # Residual connection and normalization
            node_features = vortex_out + self.layer_norms[i](attention_out)

            # If time context provided, apply harmonic processing
            if time_context is not None:
                harmonic_mod = self.harmonic_processor(node_features, time_context)
                node_features = node_features + harmonic_mod

        # Store node features for potential regularization
        self.node_features = node_features

        # Take output from central node (Node 9, index 8)
        output_features = node_features[:, 8, :]

        # Project to output dimension
        output = self.output_proj(output_features)

        return output

    def compute_regularization(self, node_features, targets=None):
        """
        Compute specialized regularization losses.

        Args:
            node_features: Node features of shape [batch_size, num_nodes, hidden_dim]
            targets: Optional target values for digital root consistency

        Returns:
            Regularization loss
        """
        from vortexnn.utils.regularization import (
            complementary_regularization_loss,
            digital_root_consistency_loss
        )

        reg_loss = complementary_regularization_loss(node_features)

        if targets is not None:
            # Get current outputs
            outputs = self.output_proj(node_features[:, 8, :])
            reg_loss += digital_root_consistency_loss(outputs, targets)

        return reg_loss

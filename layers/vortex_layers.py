"""
Standard implementations of vortex-based neural network layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vortexnn.layers.base import BaseVortexLayer


class VortexLayer(BaseVortexLayer):
    """
    Standard implementation of the Vortex Layer with enhanced stability.
    """
    def forward(self, node_features):
        """
        Forward pass for Vortex layer.

        Implementation of the update rule:
        h_i^(t) = f(W_i · AGGREGATE({h_j^(t-1) | j ∈ N(i)}) + U_i · h_i^(t-1) + bias_i)

        Args:
            node_features: tensor of shape [batch_size, num_nodes, hidden_dim]

        Returns:
            Updated node features: tensor of shape [batch_size, num_nodes, hidden_dim]
        """
        batch_size = node_features.size(0)
        device = node_features.device
        new_features = []

        # For each node i
        for i in range(self.num_nodes):
            # Get current node features
            h_i = node_features[:, i, :]  # [batch_size, hidden_dim]

            # Apply self transformation (U_i · h_i)
            self_transform = torch.matmul(h_i, self.self_weights[i])  # [batch_size, hidden_dim]

            # Aggregate messages from neighbors
            # Get indices of neighbors that send messages to node i
            neighbors = torch.nonzero(self.adjacency[i]).squeeze(-1)

            # If node has neighbors
            if neighbors.size(0) > 0:
                # Get features of all neighbors
                h_neighbors = node_features[:, neighbors, :]  # [batch_size, num_neighbors, hidden_dim]

                # Apply transformations to neighbors' messages
                neighbor_msgs = torch.zeros(batch_size, neighbors.size(0), self.hidden_dim, device=device)
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_msgs[:, j] = torch.matmul(h_neighbors[:, j], self.neighbor_weights[i])

                # Sum up all neighbor messages (AGGREGATE operation)
                neighbor_agg = torch.sum(neighbor_msgs, dim=1)  # [batch_size, hidden_dim]
            else:
                # No neighbors, so no messages
                neighbor_agg = torch.zeros(batch_size, self.hidden_dim, device=device)

            # Combine transformations and apply activation function f
            combined = self_transform + neighbor_agg + self.bias[i]

            # Apply layer normalization for stability
            normalized = self.layer_norms[i](combined)

            # Apply activation with a small slope in the negative region to prevent dead neurons
            new_h_i = F.leaky_relu(normalized, negative_slope=0.01)

            # Check for NaN values and replace if necessary
            if torch.isnan(new_h_i).any():
                new_h_i = torch.zeros_like(new_h_i)

            new_features.append(new_h_i)

        # Stack to form the output node features tensor
        return torch.stack(new_features, dim=1)


class DualFlowVortexLayer(BaseVortexLayer):
    """
    Advanced vortex layer that implements dual-flow message passing.
    Messages can flow in both clockwise and counter-clockwise directions along the doubling cycle,
    as well as between complementary pairs.
    """
    def __init__(self, num_nodes, hidden_dim):
        super(DualFlowVortexLayer, self).__init__(num_nodes, hidden_dim)

        # Additional weights for counter-clockwise flow
        self.ccw_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            for _ in range(num_nodes)
        ])

        # Weights for complementary flow
        self.comp_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            for _ in range(num_nodes)
        ])

        # Initialize additional weights
        for w in self.ccw_weights:
            nn.init.xavier_normal_(w, gain=0.5)
        for w in self.comp_weights:
            nn.init.xavier_normal_(w, gain=0.5)

        # Flow balance parameter (learnable)
        self.flow_balance = nn.Parameter(torch.tensor(0.5))

        # Define flow-specific adjacency matrices
        self._create_flow_adjacencies()

    def _create_flow_adjacencies(self):
        """
        Creates separate adjacency matrices for different flow patterns.
        """
        # Clockwise flow (following doubling cycle)
        cw_adj = torch.zeros(self.num_nodes, self.num_nodes)
        doubling_cycle = [(0, 1), (1, 3), (3, 7), (7, 6), (6, 4), (4, 0)]
        for src, dst in doubling_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                cw_adj[dst, src] = 1  # dst receives from src

        # Counter-clockwise flow (reversed doubling cycle)
        ccw_adj = torch.zeros(self.num_nodes, self.num_nodes)
        reversed_cycle = [(1, 0), (3, 1), (7, 3), (6, 7), (4, 6), (0, 4)]
        for src, dst in reversed_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                ccw_adj[dst, src] = 1  # dst receives from src

        # Complementary flow (between complementary pairs)
        comp_adj = torch.zeros(self.num_nodes, self.num_nodes)
        complementary_pairs = [(0, 7), (1, 6), (3, 4), (2, 5)]
        for a, b in complementary_pairs:
            if a < self.num_nodes and b < self.num_nodes:
                comp_adj[a, b] = comp_adj[b, a] = 1  # bidirectional

        # Register adjacencies as buffers
        self.register_buffer('cw_adjacency', cw_adj)
        self.register_buffer('ccw_adjacency', ccw_adj)
        self.register_buffer('comp_adjacency', comp_adj)

    def _process_flow(self, node_features, adjacency, weights):
        """
        Process node features through a specific flow pattern.
        """
        batch_size = node_features.size(0)
        device = node_features.device
        result = torch.zeros_like(node_features)

        for i in range(self.num_nodes):
            # Get neighbors in this flow pattern
            neighbors = torch.nonzero(adjacency[i]).squeeze(-1)

            if neighbors.size(0) > 0:
                # Get features of all neighbors
                h_neighbors = node_features[:, neighbors, :]

                # Apply transformations to neighbors' messages
                neighbor_msgs = torch.zeros(batch_size, neighbors.size(0), self.hidden_dim, device=device)
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_msgs[:, j] = torch.matmul(h_neighbors[:, j], weights[neighbor_idx])

                # Sum up all neighbor messages
                result[:, i] = torch.sum(neighbor_msgs, dim=1)

        return result

    def forward(self, node_features):
        """
        Forward pass with dual flow message passing.
        """
        # Get current flow balance
        balance = torch.sigmoid(self.flow_balance)

        # Process each flow pattern
        cw_flow = self._process_flow(node_features, self.cw_adjacency, self.neighbor_weights)
        ccw_flow = self._process_flow(node_features, self.ccw_adjacency, self.ccw_weights)
        comp_flow = self._process_flow(node_features, self.comp_adjacency, self.comp_weights)

        # Apply self-transform to each node
        self_transform = torch.zeros_like(node_features)
        for i in range(self.num_nodes):
            self_transform[:, i] = torch.matmul(node_features[:, i], self.self_weights[i])

        # Combine flows with learned balance
        combined_flow = (balance * cw_flow + (1-balance) * ccw_flow + comp_flow) + self_transform

        # Apply bias, normalization, and activation for each node
        result = torch.zeros_like(node_features)
        for i in range(self.num_nodes):
            # Add bias and normalize
            normalized = self.layer_norms[i](combined_flow[:, i] + self.bias[i])

            # Apply activation function
            result[:, i] = F.leaky_relu(normalized, negative_slope=0.01)

        return result


class HarmonicResonanceLayer(nn.Module):
    """
    Layer that applies harmonic oscillations to capture cyclical patterns in the data.
    """
    def __init__(self, hidden_dim, frequencies=[1, 2, 4, 8, 7, 5]):
        super(HarmonicResonanceLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.frequencies = frequencies
        self.oscillators = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
            for _ in frequencies
        ])
        self.phase_shifts = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim) + (i/len(frequencies))*math.pi)
            for i in range(len(frequencies))
        ])

    def forward(self, x, t=None):
        """
        Args:
            x: Input tensor of shape [batch_size, num_nodes, hidden_dim]
            t: Time or position indicator tensor of shape [batch_size, 1] normalized to [0, 2π].
               If None, uses a fixed time value.
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # If t is not provided, use a fixed value
        if t is None:
            t = torch.ones(batch_size, 1, device=x.device) * math.pi

        result = torch.zeros_like(x)

        for i in range(num_nodes):
            node_result = torch.zeros_like(x[:, i])

            for j, freq in enumerate(self.frequencies):
                # Create harmonic oscillation with learned frequency and phase
                resonator = torch.sin(freq * t + self.phase_shifts[j].unsqueeze(0))
                transformed = torch.matmul(x[:, i], self.oscillators[j]) * resonator
                node_result += transformed

            result[:, i] = node_result

        return result


class DigitalRootUnit(nn.Module):
    """
    Neural unit that processes inputs through a digital root inspired transformation.
    Digital roots (repeatedly summing digits until a single digit) are central to vortex mathematics.
    """
    def __init__(self, input_dim, hidden_dim):
        super(DigitalRootUnit, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.projection = nn.Linear(input_dim, 9 * hidden_dim)
        self.modular_weights = nn.Parameter(torch.randn(9, hidden_dim) * 0.1)

    def forward(self, x):
        # Project input to 9 separate channels (one per digit 1-9)
        batch_size = x.size(0)
        projected = self.projection(x)
        reshaped = projected.view(batch_size, 9, self.hidden_dim)

        # Apply modular weights (mimicking digital root properties)
        digital_components = reshaped * self.modular_weights.unsqueeze(0)

        # Apply digital root-inspired transformation to each node separately
        # but maintain the 9-node structure
        transformed = torch.tanh(digital_components)

        # Return tensor with shape [batch_size, 9, hidden_dim]
        return transformed

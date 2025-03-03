"""
Hyperbolic geometry based layers for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicVortexLayer(nn.Module):
    """
    Vortex layer that uses hyperbolic geometry to better represent cyclical patterns.
    """
    def __init__(self, num_nodes, hidden_dim, curvature=-1.0):
        super(HyperbolicVortexLayer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.curvature = nn.Parameter(torch.tensor(curvature))

        # Transformations between Euclidean and hyperbolic space
        self.to_hyperbolic = nn.Linear(hidden_dim, hidden_dim)
        self.from_hyperbolic = nn.Linear(hidden_dim, hidden_dim)

        # Möbius transformation weights
        self.mobius_weights = nn.Parameter(torch.Tensor(num_nodes, num_nodes, hidden_dim))

        # Initialize weights
        nn.init.kaiming_normal_(self.mobius_weights, nonlinearity='tanh')

        # Define vortex connections
        self._create_vortex_connections()

    def _create_vortex_connections(self):
        """
        Create connection matrices for the vortex structure.
        """
        # Doubling cycle connections
        doubling_adj = torch.zeros(self.num_nodes, self.num_nodes)
        doubling_cycle = [(0, 1), (1, 3), (3, 7), (7, 6), (6, 4), (4, 0)]
        for src, dst in doubling_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                doubling_adj[dst, src] = 1

        # Complementary pair connections
        comp_adj = torch.zeros(self.num_nodes, self.num_nodes)
        complementary_pairs = [(0, 7), (1, 6), (3, 4), (2, 5)]
        for a, b in complementary_pairs:
            if a < self.num_nodes and b < self.num_nodes:
                comp_adj[a, b] = comp_adj[b, a] = 1

        # Central node connections
        central_adj = torch.zeros(self.num_nodes, self.num_nodes)
        if self.num_nodes > 8:
            for i in range(8):
                central_adj[i, 8] = central_adj[8, i] = 1

        # Register adjacencies as buffers
        self.register_buffer('doubling_adjacency', doubling_adj)
        self.register_buffer('complementary_adjacency', comp_adj)
        self.register_buffer('central_adjacency', central_adj)

    def _to_poincare_ball(self, x):
        """
        Map features to the Poincaré ball model of hyperbolic space.
        """
        # Project and scale to ensure points lie within the Poincaré ball
        x_proj = self.to_hyperbolic(x)
        norm = torch.norm(x_proj, dim=-1, keepdim=True)
        return torch.tanh(norm) * (x_proj / (norm + 1e-8))

    def _from_poincare_ball(self, x):
        """
        Map from Poincaré ball back to Euclidean space.
        """
        # Project from hyperbolic to Euclidean space
        return self.from_hyperbolic(x)

    def _mobius_addition(self, x, y):
        """
        Addition in the Poincaré ball model of hyperbolic space.
        """
        # Compute Möbius addition formula
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y**2, dim=-1, keepdim=True)

        c = self.curvature.abs()  # Use absolute value for numerical stability

        numerator = (1 + 2*c*xy_dot + c*y_norm_sq)*x + (1 - c*x_norm_sq)*y
        denominator = 1 + 2*c*xy_dot + c**2*x_norm_sq*y_norm_sq

        return numerator / (denominator + 1e-8)

    def forward(self, node_features):
        """
        Forward pass using hyperbolic operations.
        """
        batch_size = node_features.size(0)

        # Map to hyperbolic space
        hyperbolic_features = self._to_poincare_ball(node_features)

        # Initialize results with self features
        result = hyperbolic_features.clone()

        # Process each node
        for i in range(self.num_nodes):
            node_result = hyperbolic_features[:, i]

            # Process doubling cycle connections
            doubling_neighbors = torch.nonzero(self.doubling_adjacency[i]).squeeze(-1)
            for j in doubling_neighbors:
                # Apply Möbius transformation
                transformed = self._mobius_addition(
                    hyperbolic_features[:, j],
                    self.mobius_weights[i, j].unsqueeze(0)
                )
                node_result = self._mobius_addition(node_result, transformed)

            # Process complementary connections
            comp_neighbors = torch.nonzero(self.complementary_adjacency[i]).squeeze(-1)
            for j in comp_neighbors:
                transformed = self._mobius_addition(
                    hyperbolic_features[:, j],
                    self.mobius_weights[i, j].unsqueeze(0)
                )
                node_result = self._mobius_addition(node_result, transformed)

            # Process central node connections
            if self.num_nodes > 8:
                central_neighbors = torch.nonzero(self.central_adjacency[i]).squeeze(-1)
                for j in central_neighbors:
                    transformed = self._mobius_addition(
                        hyperbolic_features[:, j],
                        self.mobius_weights[i, j].unsqueeze(0)
                    )
                    node_result = self._mobius_addition(node_result, transformed)

            result[:, i] = node_result

        # Map back to Euclidean space
        return self._from_poincare_ball(result)

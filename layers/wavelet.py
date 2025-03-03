"""
Wavelet-based and modular field layers for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletVortexLayer(nn.Module):
    """
    Vortex layer that uses wavelet-based multi-resolution analysis to capture patterns at different scales.
    """
    def __init__(self, num_nodes, hidden_dim, levels=3):
        super(WaveletVortexLayer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.levels = levels

        # Learnable wavelet filters
        self.dec_filters = nn.Parameter(torch.Tensor(levels, 4))
        self.rec_filters = nn.Parameter(torch.Tensor(levels, 4))

        # Initialize with approximate Daubechies wavelet coefficients
        self._initialize_wavelet_filters()

        # Node-specific transformations
        self.node_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_nodes)
        ])

        # Normalization and activation
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_nodes)
        ])

        # Define scale mapping based on doubling cycle
        self.scale_map = {0: 0, 1: 1, 3: 2, 7: 2, 6: 1, 4: 0, 2: 1, 5: 1, 8: 0}

    def _initialize_wavelet_filters(self):
        """
        Initialize wavelet filters with approximate Daubechies coefficients.
        """
        # Approximate Daubechies 2 wavelet coefficients
        h0 = torch.tensor([0.6830, 1.1830, 0.3170, -0.1830])
        h1 = torch.tensor([-0.1830, -0.3170, 1.1830, -0.6830])

        # Initialize all levels with variations of these coefficients
        for i in range(self.levels):
            # Add some variation for different levels
            scale = 1.0 - 0.1 * i
            self.dec_filters.data[i] = h0 * scale
            self.rec_filters.data[i] = h1 * scale

    def _wavelet_transform(self, x, level):
        """
        Apply wavelet decomposition and reconstruction.
        """
        # Get filters for this level
        h0 = self.dec_filters[level]
        h1 = self.rec_filters[level]

        # Handle multi-channel input for wavelet transform
        batch_size, hidden_dim = x.shape

        # Process each channel separately
        result = torch.zeros_like(x)

        # This is a simplified implementation - in practice, you'd use a proper wavelet library
        for i in range(hidden_dim):
            # Extract single channel and add dummy dimensions for conv1d
            channel = x[:, i].unsqueeze(1).unsqueeze(1)

            # Add padding
            padded = F.pad(channel, (1, 2))

            # Apply filters
            approx = F.conv1d(padded, h0.view(1, 1, -1))
            detail = F.conv1d(padded, h1.view(1, 1, -1))

            # Combine and remove dummy dimension
            result[:, i] = (approx + detail).squeeze(1).squeeze(1)

        return result

    def forward(self, node_features):
        """
        Forward pass using wavelet transformations.
        """
        batch_size = node_features.size(0)
        result = torch.zeros_like(node_features)

        for i in range(self.num_nodes):
            # Get appropriate wavelet scale for this node
            scale = self.scale_map.get(i, 0) % self.levels

            # Apply wavelet transform
            wavelet_features = self._wavelet_transform(node_features[:, i], scale)

            # Apply node-specific transformation
            transformed = self.node_transforms[i](wavelet_features)

            # Normalize and activate
            normalized = self.norms[i](transformed)
            result[:, i] = F.leaky_relu(normalized, negative_slope=0.01)

        return result


class ModularFieldVortexLayer(nn.Module):
    """
    Vortex layer that uses modular field algebra to formalize digital root operations.
    """
    def __init__(self, num_nodes, hidden_dim, modulus=9):
        super(ModularFieldVortexLayer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.modulus = modulus

        # Field transformations for each node and modular class
        self.field_transforms = nn.ParameterList([
            nn.Parameter(torch.Tensor(modulus, hidden_dim, hidden_dim))
            for _ in range(num_nodes)
        ])

        # Initialize transformations
        for transforms in self.field_transforms:
            for i in range(modulus):
                nn.init.orthogonal_(transforms[i])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_nodes)
        ])

        # Precompute digital root mapping
        self._create_digital_root_map()

        # Define vortex connections
        self._create_vortex_adjacency()

    def _create_digital_root_map(self):
        """
        Precompute digital root mapping for efficient lookup.
        """
        mapping = torch.zeros(100, dtype=torch.long)
        for i in range(100):
            # Digital root: sum digits until single digit
            n = i
            while n >= 10:
                n = sum(int(digit) for digit in str(n))
            mapping[i] = n if n > 0 else 9  # 9 instead of 0

        self.register_buffer('digital_root_map', mapping)

    def _create_vortex_adjacency(self):
        """
        Create adjacency matrix for the vortex structure.
        """
        adj = torch.zeros(self.num_nodes, self.num_nodes)

        # Doubling Cycle
        doubling_cycle = [(0, 1), (1, 3), (3, 7), (7, 6), (6, 4), (4, 0)]
        for src, dst in doubling_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                adj[dst, src] = 1

        # Complementary Pairs
        complementary_pairs = [(0, 7), (7, 0), (1, 6), (6, 1), (3, 4), (4, 3), (2, 5), (5, 2)]
        for src, dst in complementary_pairs:
            if src < self.num_nodes and dst < self.num_nodes:
                adj[dst, src] = 1

        # Central Node
        if self.num_nodes > 8:
            for i in range(8):
                adj[i, 8] = adj[8, i] = 1

        self.register_buffer('adjacency', adj)

    def _get_digital_root(self, x):
        """
        Compute approximate digital root of feature tensor.
        """
        # Scale values to [0, 99] range for lookup table
        x_scaled = torch.clamp((x + 1) * 49, 0, 99).long()

        # Get digital roots (simplified approximation)
        return self.digital_root_map[x_scaled]

    def forward(self, node_features):
        """
        Forward pass using modular field operations.
        """
        batch_size = node_features.size(0)
        device = node_features.device
        result = torch.zeros_like(node_features)

        for i in range(self.num_nodes):
            # Compute digital root signature for this node's features
            feature_sum = torch.sum(node_features[:, i], dim=-1)
            digital_root = torch.fmod(torch.abs(feature_sum), self.modulus.float()).long()
            digital_root = torch.where(digital_root == 0, torch.tensor(9, device=device), digital_root)

            # Apply modular field transformation based on digital root
            transform_weights = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim, device=device)
            for b in range(batch_size):
                root_idx = digital_root[b].item() - 1  # Convert to 0-indexed
                transform_weights[b] = self.field_transforms[i][root_idx]

            # Apply transformation
            node_result = torch.zeros(batch_size, self.hidden_dim, device=device)
            for b in range(batch_size):
                node_result[b] = torch.matmul(node_features[b, i], transform_weights[b])

            # Process neighbors according to vortex structure
            neighbors = torch.nonzero(self.adjacency[i]).squeeze(-1)
            if neighbors.size(0) > 0:
                neighbor_result = torch.zeros_like(node_result)
                for j in neighbors:
                    # Get digital root of neighbor
                    neighbor_sum = torch.sum(node_features[:, j], dim=-1)
                    neighbor_root = torch.fmod(torch.abs(neighbor_sum), self.modulus.float()).long()
                    neighbor_root = torch.where(neighbor_root == 0, torch.tensor(9, device=device), neighbor_root)

                    # Apply modular field transformation
                    for b in range(batch_size):
                        root_idx = neighbor_root[b].item() - 1
                        transform = self.field_transforms[i][root_idx]
                        neighbor_result[b] += torch.matmul(node_features[b, j], transform)

                # Combine self and neighbor results
                node_result = node_result + neighbor_result

            # Apply normalization and activation
            normalized = self.layer_norms[i](node_result)
            result[:, i] = F.leaky_relu(normalized, negative_slope=0.01)

        return result

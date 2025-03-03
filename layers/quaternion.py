"""
Quaternion-based layers for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vortexnn.layers.base import BaseVortexLayer


class QuaternionLinear(nn.Module):
    """
    Linear layer using quaternion algebra for non-commutative operations.
    """
    def __init__(self, in_features, out_features):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Ensure dimensions are divisible by 4 (for quaternion representation)
        assert in_features % 4 == 0, "Input features must be divisible by 4"
        assert out_features % 4 == 0, "Output features must be divisible by 4"

        in_q = in_features // 4
        out_q = out_features // 4

        # Create weight matrices for each quaternion component
        self.Wa = nn.Parameter(torch.Tensor(in_q, out_q))
        self.Wb = nn.Parameter(torch.Tensor(in_q, out_q))
        self.Wc = nn.Parameter(torch.Tensor(in_q, out_q))
        self.Wd = nn.Parameter(torch.Tensor(in_q, out_q))

        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights following quaternion initialization scheme.
        """
        nn.init.xavier_uniform_(self.Wa)
        nn.init.xavier_uniform_(self.Wb)
        nn.init.xavier_uniform_(self.Wc)
        nn.init.xavier_uniform_(self.Wd)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        Apply quaternion linear transformation.
        """
        batch_size = input.size(0)

        # Split input into quaternion components
        in_size = self.in_features // 4
        a, b, c, d = input.view(batch_size, -1).chunk(4, dim=1)

        # Reshape for matrix multiplication
        a = a.view(batch_size, -1, in_size)
        b = b.view(batch_size, -1, in_size)
        c = c.view(batch_size, -1, in_size)
        d = d.view(batch_size, -1, in_size)

        # Hamilton product for matrix multiplication
        out_a = torch.matmul(a, self.Wa) - torch.matmul(b, self.Wb) - torch.matmul(c, self.Wc) - torch.matmul(d, self.Wd)
        out_b = torch.matmul(a, self.Wb) + torch.matmul(b, self.Wa) + torch.matmul(c, self.Wd) - torch.matmul(d, self.Wc)
        out_c = torch.matmul(a, self.Wc) - torch.matmul(b, self.Wd) + torch.matmul(c, self.Wa) + torch.matmul(d, self.Wb)
        out_d = torch.matmul(a, self.Wd) + torch.matmul(b, self.Wc) - torch.matmul(c, self.Wb) + torch.matmul(d, self.Wa)

        # Reshape outputs
        out_a = out_a.view(batch_size, -1)
        out_b = out_b.view(batch_size, -1)
        out_c = out_c.view(batch_size, -1)
        out_d = out_d.view(batch_size, -1)

        # Concatenate outputs and add bias
        return torch.cat([out_a, out_b, out_c, out_d], dim=1) + self.bias


class QuaternionVortexLayer(BaseVortexLayer):
    """
    Vortex layer that uses quaternion algebra for non-commutative operations.
    """
    def __init__(self, num_nodes, hidden_dim):
        super(QuaternionVortexLayer, self).__init__(num_nodes, hidden_dim)

        # Ensure hidden_dim is divisible by 4 for quaternion representation
        assert hidden_dim % 4 == 0, "Hidden dimension must be divisible by 4 for quaternion operations"

        # Override the standard weights with quaternion weights
        self.quaternion_transforms = nn.ModuleList([
            QuaternionLinear(hidden_dim, hidden_dim) for _ in range(num_nodes)
        ])

        # Quaternion normalization
        self.q_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_nodes)
        ])

    def forward(self, node_features):
        """
        Forward pass using quaternion operations.
        """
        batch_size = node_features.size(0)
        device = node_features.device
        new_features = []

        # For each node i
        for i in range(self.num_nodes):
            # Apply quaternion transformation to self
            self_result = self.quaternion_transforms[i](node_features[:, i])

            # Aggregate messages from neighbors
            neighbors = torch.nonzero(self.adjacency[i]).squeeze(-1)

            if neighbors.size(0) > 0:
                # Get features of all neighbors
                h_neighbors = node_features[:, neighbors, :]

                # Sum of transformed neighbors
                neighbor_sum = torch.zeros(batch_size, self.hidden_dim, device=device)
                for j, neighbor_idx in enumerate(neighbors):
                    # Apply quaternion transformation
                    neighbor_sum += self.quaternion_transforms[neighbor_idx](h_neighbors[:, j])

                # Combine self and neighbor features
                combined = self_result + neighbor_sum
            else:
                combined = self_result

            # Apply normalization and activation
            normalized = self.q_norms[i](combined)
            activated = F.leaky_relu(normalized, negative_slope=0.01)

            new_features.append(activated)

        # Stack to form the output node features tensor
        return torch.stack(new_features, dim=1)

"""
Attention mechanisms for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VortexAttention(nn.Module):
    """
    Attention mechanism specialized for the vortex structure, allowing nodes to
    selectively attend to different parts of the vortex graph.
    """
    def __init__(self, num_nodes, hidden_dim):
        super(VortexAttention, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Query, key, value projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Position encodings based on vortex positions (1-9)
        self.position_encodings = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.1)

        # For special handling of central node
        if num_nodes > 8:
            self.central_gate = nn.Linear(hidden_dim, 1)

        # Create vortex-specific attention mask
        self._create_vortex_attention_mask()

    def _create_vortex_attention_mask(self):
        """
        Creates an attention mask that encodes the vortex structure.
        """
        mask = torch.ones(self.num_nodes, self.num_nodes)

        # Enhance doubling cycle connections
        doubling_cycle = [(0, 1), (1, 3), (3, 7), (7, 6), (6, 4), (4, 0)]
        for src, dst in doubling_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                mask[src, dst] = mask[dst, src] = 2.0

        # Enhance complementary pair connections
        complementary_pairs = [(0, 7), (1, 6), (3, 4), (2, 5)]
        for a, b in complementary_pairs:
            if a < self.num_nodes and b < self.num_nodes:
                mask[a, b] = mask[b, a] = 2.0

        # Enhance central node connections
        if self.num_nodes > 8:
            mask[8, :] = mask[:, 8] = 1.5

        self.register_buffer('attention_mask', mask)

    def forward(self, node_features):
        """
        Forward pass applying vortex-structured attention.
        """
        batch_size = node_features.size(0)

        # Add position encodings
        positioned_features = node_features + self.position_encodings.unsqueeze(0)

        # Project to queries, keys, values
        queries = self.query_proj(positioned_features)
        keys = self.key_proj(positioned_features)
        values = self.value_proj(positioned_features)

        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

        # Apply vortex-specific attention mask
        attention_scores = attention_scores * self.attention_mask

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attention_output = torch.matmul(attention_weights, values)

        # Special handling for central node
        if self.num_nodes > 8 and hasattr(self, 'central_gate'):
            central_importance = torch.sigmoid(self.central_gate(node_features[:, 8])).unsqueeze(1)
            central_influence = torch.zeros_like(attention_output)
            central_influence += node_features[:, 8].unsqueeze(1)
            attention_output = attention_output * (1 - central_importance) + central_influence * central_importance

        return attention_output

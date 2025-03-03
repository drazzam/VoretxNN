"""
Base layer definitions for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseVortexLayer(nn.Module):
    """
    Base layer for vortex-inspired graph neural network.
    Implements the core message passing mechanism based on the vortex mathematics structure.
    """
    def __init__(self, num_nodes, hidden_dim):
        super(BaseVortexLayer, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Create weight matrices for message passing
        # Each node has its own set of weight matrices

        # W_i matrices for neighbor aggregation
        self.neighbor_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            for _ in range(num_nodes)
        ])

        # U_i matrices for self loops
        self.self_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            for _ in range(num_nodes)
        ])

        # Bias terms
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim))
            for _ in range(num_nodes)
        ])

        # Layer normalization for each node to improve training stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_nodes)
        ])

        # Initialize weights
        for w in self.neighbor_weights:
            nn.init.xavier_normal_(w, gain=0.5)
        for w in self.self_weights:
            nn.init.xavier_normal_(w, gain=0.5)
        for b in self.bias:
            nn.init.zeros_(b)

        # Define the vortex graph structure - will be used by inheriting classes
        self._create_vortex_adjacency()

    def _create_vortex_adjacency(self):
        """
        Creates the adjacency matrix based on vortex mathematics patterns.
        The adjacency matrix defines which nodes can send messages to which other nodes.
        adjacency[i,j] = 1 means node i receives a message from node j.
        """
        adj = torch.zeros(self.num_nodes, self.num_nodes)

        # Doubling Cycle: 1→2→4→8→7→5→1 (directed)
        # Note: we use 0-based indexing, so node 1 is index 0, etc.
        doubling_cycle = [(0, 1), (1, 3), (3, 7), (7, 6), (6, 4), (4, 0)]
        for src, dst in doubling_cycle:
            if src < self.num_nodes and dst < self.num_nodes:
                adj[dst, src] = 1  # dst receives from src

        # Complementary Pairs: 1↔8, 2↔7, 4↔5, 3↔6 (bidirectional)
        complementary_pairs = [(0, 7), (7, 0), (1, 6), (6, 1), (3, 4), (4, 3), (2, 5), (5, 2)]
        for src, dst in complementary_pairs:
            if src < self.num_nodes and dst < self.num_nodes:
                adj[dst, src] = 1  # dst receives from src

        # Central Node 9 (index 8): connects to all other nodes (bidirectional)
        if self.num_nodes > 8:
            for i in range(8):
                adj[i, 8] = 1  # i receives from 9
                adj[8, i] = 1  # 9 receives from i

        # Register as buffer so it moves to the correct device with the model
        self.register_buffer('adjacency', adj)

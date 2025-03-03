"""
Data generation utilities for Vortex Neural Networks.
These functions generate synthetic data that exhibits vortex mathematics patterns.
"""

import numpy as np


def generate_cyclical_data(n_samples=1000, input_dim=9, seed=42):
    """
    Generate data with cyclical patterns based on vortex mathematics.

    Args:
        n_samples: Number of samples to generate
        input_dim: Input dimension (should be at least 9 for vortex structure)
        seed: Random seed for reproducibility

    Returns:
        X: Features
        y: Target values
    """
    np.random.seed(seed)

    # Create features
    X = np.zeros((n_samples, input_dim))

    # Time steps for sine waves
    t = np.linspace(0, 4*np.pi, n_samples)

    # Create features with different sine wave frequencies
    for i in range(input_dim):
        freq = (i + 1) / 3.0
        X[:, i] = np.sin(freq * t) + 0.1 * np.random.randn(n_samples)

    # The doubling cycle in vortex mathematics (0-indexed)
    doubling_cycle = [0, 1, 3, 7, 6, 4]  # 1→2→4→8→7→5 (0-indexed)

    # Set up weights: stronger for doubling cycle indices
    weights = np.ones(input_dim) * 0.2
    for i in doubling_cycle:
        if i < input_dim:
            weights[i] = 1.0  # Stronger weight for doubling cycle

    # Create target as weighted sum, emphasizing the doubling cycle pattern
    y = np.zeros(n_samples)

    # Add cyclical interactions between consecutive nodes in the doubling cycle
    for i in range(len(doubling_cycle)):
        idx1 = doubling_cycle[i]
        idx2 = doubling_cycle[(i + 1) % len(doubling_cycle)]
        if idx1 < input_dim and idx2 < input_dim:
            y += weights[idx1] * X[:, idx1] + weights[idx2] * X[:, idx2] + 0.5 * X[:, idx1] * X[:, idx2]

    # Add non-linear transformation with modulo operation (related to vortex mathematics)
    y = np.sin(y) + 0.3 * np.cos(y) + 0.2 * np.sin(y) * np.cos(y)

    # Add noise
    y += 0.1 * np.random.randn(n_samples)

    return X, y.reshape(-1, 1)


def generate_complementary_data(n_samples=1000, input_dim=9, seed=43):
    """
    Generate data with complementary pair relationships from vortex mathematics.

    Args:
        n_samples: Number of samples to generate
        input_dim: Input dimension (should be at least 9 for vortex structure)
        seed: Random seed for reproducibility

    Returns:
        X: Features
        y: Target values
    """
    np.random.seed(seed)

    # Create features
    X = np.random.randn(n_samples, input_dim)

    # Define complementary pairs (0-indexed)
    complementary_pairs = [(0, 7), (1, 6), (3, 4), (2, 5)]  # 1-8, 2-7, 4-5, 3-6

    # Create target based on complementary pair interactions
    y = np.zeros(n_samples)

    for pair in complementary_pairs:
        if pair[0] < input_dim and pair[1] < input_dim:
            # Create a non-linear interaction between complementary pairs
            interaction = np.sin(X[:, pair[0]] * X[:, pair[1]])

            # Add to target
            y += interaction

    # Add a central node effect (node 9)
    if input_dim > 8:
        central_effect = 0.5 * np.mean(X[:, :8], axis=1) + 0.2 * X[:, 8]
        y += central_effect

    # Add noise
    y += 0.1 * np.random.randn(n_samples)

    return X, y.reshape(-1, 1)

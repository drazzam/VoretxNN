"""
Custom activation functions and normalization layers for Vortex Neural Networks.
"""

import torch
import torch.nn as nn
import math


class ModularActivation(nn.Module):
    """
    Activation function based on modular arithmetic principles from vortex mathematics.
    """
    def __init__(self, base=9, trainable=True):
        super(ModularActivation, self).__init__()
        self.base = base
        # Trainable coefficients for a weighted sum of modular basis functions
        if trainable:
            self.coefficients = nn.Parameter(torch.ones(base))
        else:
            self.register_buffer('coefficients', torch.ones(base))

    def forward(self, x):
        """
        Apply modular activation function.
        """
        # Create a series of modular basis functions
        result = torch.zeros_like(x)
        for i in range(1, self.base+1):
            # Each basis function has a period related to i
            basis = torch.sin(x * i * math.pi / self.base)
            result += self.coefficients[i-1] * basis

        return result


class DigitalRootNorm(nn.Module):
    """
    Normalization layer that incorporates digital root properties from vortex mathematics.
    """
    def __init__(self, hidden_dim, epsilon=1e-5):
        super(DigitalRootNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.epsilon = epsilon

    def forward(self, x):
        """
        Apply digital root normalization.
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        # Standard normalization
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)

        # Digital root transformation (project to range [1,9])
        x_mod9 = 4.0 * torch.tanh(x_norm / 4.0) + 5.0

        # Scale and shift
        return self.gamma * x_mod9 + self.beta

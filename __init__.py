"""
VortexNN: Advanced Vortex-inspired Graph Neural Network

A deep learning library based on vortex mathematics principles, offering
specialized layers, models, and utilities for pattern recognition and time series analysis.
"""

__version__ = "0.1.0"

# Import key components to make them accessible directly from the package
from vortexnn.models.advanced_vortex_gnn import AdvancedVortexGNN
from vortexnn.utils.train import train_vortex_model
from vortexnn.utils.evaluate import evaluate_vortex_model
from vortexnn.utils.data import generate_cyclical_data, generate_complementary_data
from vortexnn.utils.scheduler import VortexCyclicalLR

# Define what's accessible via the public API
__all__ = [
    'AdvancedVortexGNN',
    'train_vortex_model',
    'evaluate_vortex_model',
    'generate_cyclical_data',
    'generate_complementary_data',
    'VortexCyclicalLR',
]

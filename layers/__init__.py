"""
Vortex Neural Network Layers.

This subpackage provides specialized layers for building vortex-inspired neural networks.
"""

# Import core layers
from vortexnn.layers.base import BaseVortexLayer
from vortexnn.layers.vortex_layers import VortexLayer, DualFlowVortexLayer, HarmonicResonanceLayer, DigitalRootUnit
from vortexnn.layers.attention import VortexAttention
from vortexnn.layers.activation import ModularActivation, DigitalRootNorm
from vortexnn.layers.hyperbolic import HyperbolicVortexLayer
from vortexnn.layers.quaternion import QuaternionLinear, QuaternionVortexLayer
from vortexnn.layers.wavelet import WaveletVortexLayer, ModularFieldVortexLayer

# Define public API
__all__ = [
    'BaseVortexLayer',
    'VortexLayer',
    'DualFlowVortexLayer',
    'HarmonicResonanceLayer',
    'DigitalRootUnit',
    'VortexAttention',
    'ModularActivation',
    'DigitalRootNorm',
    'HyperbolicVortexLayer',
    'QuaternionLinear',
    'QuaternionVortexLayer',
    'WaveletVortexLayer',
    'ModularFieldVortexLayer',
]

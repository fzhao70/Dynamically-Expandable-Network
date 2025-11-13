"""
Dynamically Expandable Network (DEN)
A PyTorch implementation of neural networks that can grow during training.
"""

from .core import DynamicExpandableNetwork
from .layers import ExpandableLinear
from .trainer import DENTrainer
from .growth_strategy import (
    GrowthStrategy,
    LossBasedGrowth,
    GradientBasedGrowth,
    AdaptiveGrowth,
    BiologicalGrowth
)

__version__ = "0.1.0"
__all__ = [
    "DynamicExpandableNetwork",
    "ExpandableLinear",
    "DENTrainer",
    "GrowthStrategy",
    "LossBasedGrowth",
    "GradientBasedGrowth",
    "AdaptiveGrowth",
    "BiologicalGrowth",
]

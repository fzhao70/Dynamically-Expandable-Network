"""
Growth strategies for determining when and how to expand the network.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod


class GrowthStrategy(ABC):
    """
    Abstract base class for growth strategies.

    Growth strategies decide when and how the network should expand
    based on training metrics and network state.
    """

    def __init__(self, patience: int = 10, cooldown: int = 5):
        """
        Args:
            patience: Number of epochs to wait before triggering growth
            cooldown: Number of epochs to wait after growth before allowing another growth
        """
        self.patience = patience
        self.cooldown = cooldown
        self.epochs_since_improvement = 0
        self.epochs_since_growth = 0
        self.best_metric = float('inf')
        self.growth_triggered = False

    @abstractmethod
    def should_grow(
        self,
        metrics: Dict[str, float],
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if the network should grow.

        Args:
            metrics: Dictionary of training metrics (loss, gradient_norm, etc.)
            network: The DynamicExpandableNetwork instance
            epoch: Current epoch number

        Returns:
            Tuple of (should_grow, reason)
        """
        pass

    @abstractmethod
    def determine_growth_action(
        self,
        metrics: Dict[str, float],
        network: Any
    ) -> Dict[str, Any]:
        """
        Determine what type of growth to perform.

        Args:
            metrics: Dictionary of training metrics
            network: The DynamicExpandableNetwork instance

        Returns:
            Dictionary with growth action details:
            {
                'type': 'width' or 'depth',
                'layer_idx': int (for width expansion),
                'position': int (for depth expansion),
                'num_neurons': int
            }
        """
        pass

    def reset(self):
        """Reset the strategy state."""
        self.epochs_since_improvement = 0
        self.epochs_since_growth = 0
        self.best_metric = float('inf')
        self.growth_triggered = False


class LossBasedGrowth(GrowthStrategy):
    """
    Growth strategy based on loss plateaus.

    Triggers growth when the loss stops improving for a certain number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        cooldown: int = 5,
        min_delta: float = 1e-4,
        width_growth_ratio: float = 0.5,
        depth_threshold: int = 3,
        max_neurons_per_expansion: int = 32
    ):
        """
        Args:
            patience: Epochs to wait for improvement before growing
            cooldown: Epochs to wait after growth
            min_delta: Minimum change to be considered an improvement
            width_growth_ratio: Ratio of neurons to add (relative to current layer size)
            depth_threshold: Add depth after this many width expansions
            max_neurons_per_expansion: Maximum neurons to add in one expansion
        """
        super().__init__(patience, cooldown)
        self.min_delta = min_delta
        self.width_growth_ratio = width_growth_ratio
        self.depth_threshold = depth_threshold
        self.max_neurons_per_expansion = max_neurons_per_expansion
        self.width_expansions = 0

    def should_grow(
        self,
        metrics: Dict[str, float],
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if we should grow based on loss plateau."""
        current_loss = metrics.get('loss', float('inf'))

        # Check if we're in cooldown period
        if self.epochs_since_growth < self.cooldown:
            self.epochs_since_growth += 1
            return False, None

        # Check for improvement
        if current_loss < (self.best_metric - self.min_delta):
            self.best_metric = current_loss
            self.epochs_since_improvement = 0
            return False, None

        # No improvement
        self.epochs_since_improvement += 1

        # Check if we should grow
        if self.epochs_since_improvement >= self.patience:
            reason = f"Loss plateaued for {self.patience} epochs (current: {current_loss:.6f}, best: {self.best_metric:.6f})"
            self.epochs_since_improvement = 0
            self.epochs_since_growth = 0
            return True, reason

        return False, None

    def determine_growth_action(
        self,
        metrics: Dict[str, float],
        network: Any
    ) -> Dict[str, Any]:
        """Determine whether to expand width or depth."""
        # Alternate between width and depth expansion
        if self.width_expansions >= self.depth_threshold and len(network.layers) < 10:
            # Add depth
            self.width_expansions = 0

            # Add layer in the middle
            position = len(network.layers) // 2
            num_neurons = min(
                int(np.mean(network.get_layer_sizes())),
                self.max_neurons_per_expansion
            )

            return {
                'type': 'depth',
                'position': position,
                'num_neurons': num_neurons
            }
        else:
            # Expand width - choose layer with smallest relative size
            layer_sizes = network.get_layer_sizes()
            layer_idx = np.argmin(layer_sizes)

            # Calculate neurons to add
            current_size = layer_sizes[layer_idx]
            num_neurons = min(
                max(int(current_size * self.width_growth_ratio), 1),
                self.max_neurons_per_expansion
            )

            self.width_expansions += 1

            return {
                'type': 'width',
                'layer_idx': layer_idx,
                'num_neurons': num_neurons
            }


class GradientBasedGrowth(GrowthStrategy):
    """
    Growth strategy based on gradient magnitudes.

    Monitors gradient flow and expands layers with large gradients,
    indicating they might be under-parameterized.
    """

    def __init__(
        self,
        patience: int = 10,
        cooldown: int = 5,
        gradient_threshold: float = 0.1,
        min_delta: float = 1e-4,
        max_neurons_per_expansion: int = 32
    ):
        """
        Args:
            patience: Epochs to wait before growing
            cooldown: Epochs after growth
            gradient_threshold: Threshold for gradient magnitude
            min_delta: Minimum improvement threshold
            max_neurons_per_expansion: Max neurons to add
        """
        super().__init__(patience, cooldown)
        self.gradient_threshold = gradient_threshold
        self.min_delta = min_delta
        self.max_neurons_per_expansion = max_neurons_per_expansion
        self.gradient_history = []

    def should_grow(
        self,
        metrics: Dict[str, float],
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """Check if we should grow based on gradients and loss."""
        current_loss = metrics.get('loss', float('inf'))
        avg_gradient = metrics.get('avg_gradient_norm', 0.0)

        # Store gradient history
        self.gradient_history.append(avg_gradient)
        if len(self.gradient_history) > self.patience:
            self.gradient_history.pop(0)

        # Check cooldown
        if self.epochs_since_growth < self.cooldown:
            self.epochs_since_growth += 1
            return False, None

        # Check for improvement
        if current_loss < (self.best_metric - self.min_delta):
            self.best_metric = current_loss
            self.epochs_since_improvement = 0
            return False, None

        self.epochs_since_improvement += 1

        # Check if gradients are large and loss is not improving
        if len(self.gradient_history) >= self.patience:
            avg_recent_grad = np.mean(self.gradient_history)

            if avg_recent_grad > self.gradient_threshold and \
               self.epochs_since_improvement >= self.patience:
                reason = f"High gradients ({avg_recent_grad:.6f}) with loss plateau"
                self.epochs_since_improvement = 0
                self.epochs_since_growth = 0
                self.gradient_history = []
                return True, reason

        return False, None

    def determine_growth_action(
        self,
        metrics: Dict[str, float],
        network: Any
    ) -> Dict[str, Any]:
        """Expand the layer with highest gradient magnitude."""
        layer_gradients = metrics.get('layer_gradients', [])

        if layer_gradients:
            # Expand layer with largest gradient
            layer_idx = np.argmax(layer_gradients)
        else:
            # Fallback: expand smallest layer
            layer_sizes = network.get_layer_sizes()
            layer_idx = np.argmin(layer_sizes)

        current_size = network.layers[layer_idx].out_features
        num_neurons = min(
            max(int(current_size * 0.3), 1),
            self.max_neurons_per_expansion
        )

        return {
            'type': 'width',
            'layer_idx': layer_idx,
            'num_neurons': num_neurons
        }


class AdaptiveGrowth(GrowthStrategy):
    """
    Adaptive growth strategy that combines multiple signals.

    Uses loss, gradients, and network capacity to make growth decisions.
    """

    def __init__(
        self,
        patience: int = 15,
        cooldown: int = 10,
        loss_threshold: float = 1e-4,
        gradient_threshold: float = 0.05,
        capacity_threshold: float = 0.8,
        max_neurons_per_expansion: int = 64,
        max_network_size: int = 10000
    ):
        """
        Args:
            patience: Epochs to wait
            cooldown: Epochs after growth
            loss_threshold: Loss improvement threshold
            gradient_threshold: Gradient magnitude threshold
            capacity_threshold: Network capacity utilization threshold
            max_neurons_per_expansion: Max neurons per expansion
            max_network_size: Maximum total network parameters
        """
        super().__init__(patience, cooldown)
        self.loss_threshold = loss_threshold
        self.gradient_threshold = gradient_threshold
        self.capacity_threshold = capacity_threshold
        self.max_neurons_per_expansion = max_neurons_per_expansion
        self.max_network_size = max_network_size
        self.loss_history = []

    def should_grow(
        self,
        metrics: Dict[str, float],
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """Adaptive growth decision based on multiple factors."""
        current_loss = metrics.get('loss', float('inf'))

        # Check if network is already too large
        if network.get_num_parameters() >= self.max_network_size:
            return False, None

        # Track loss
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.patience:
            self.loss_history.pop(0)

        # Cooldown check
        if self.epochs_since_growth < self.cooldown:
            self.epochs_since_growth += 1
            return False, None

        # Update best metric
        if current_loss < self.best_metric:
            self.best_metric = current_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Need sufficient history
        if len(self.loss_history) < self.patience:
            return False, None

        # Calculate loss improvement rate
        loss_std = np.std(self.loss_history)
        loss_improvement = self.loss_history[0] - self.loss_history[-1]

        # Get gradient info
        avg_gradient = metrics.get('avg_gradient_norm', 0.0)

        # Decision criteria
        loss_stagnant = loss_improvement < self.loss_threshold and loss_std < self.loss_threshold
        high_gradients = avg_gradient > self.gradient_threshold

        if (loss_stagnant or high_gradients) and self.epochs_since_improvement >= self.patience:
            reasons = []
            if loss_stagnant:
                reasons.append(f"loss stagnant (improvement: {loss_improvement:.6f})")
            if high_gradients:
                reasons.append(f"high gradients ({avg_gradient:.6f})")

            self.epochs_since_improvement = 0
            self.epochs_since_growth = 0
            self.loss_history = []
            return True, ", ".join(reasons)

        return False, None

    def determine_growth_action(
        self,
        metrics: Dict[str, float],
        network: Any
    ) -> Dict[str, Any]:
        """Intelligently choose growth type and location."""
        layer_sizes = network.get_layer_sizes()
        num_layers = len(layer_sizes)

        # Get layer-wise metrics if available
        layer_gradients = metrics.get('layer_gradients', [0] * num_layers)
        layer_activations = metrics.get('layer_activations', [0] * num_layers)

        # Calculate a score for each layer
        scores = []
        for i, (size, grad, act) in enumerate(zip(layer_sizes, layer_gradients, layer_activations)):
            # Prefer layers with high gradients, low size, and moderate activations
            score = grad / (size + 1) * (1.0 + act)
            scores.append(score)

        # Decide between width and depth
        avg_layer_size = np.mean(layer_sizes)
        size_variance = np.var(layer_sizes)

        # Add depth if network is shallow or layers are very uniform
        if num_layers < 3 or (size_variance < avg_layer_size * 0.1 and num_layers < 8):
            position = num_layers // 2
            num_neurons = min(
                int(avg_layer_size),
                self.max_neurons_per_expansion
            )
            return {
                'type': 'depth',
                'position': position,
                'num_neurons': num_neurons
            }
        else:
            # Expand width of layer with highest score
            layer_idx = np.argmax(scores)
            current_size = layer_sizes[layer_idx]
            num_neurons = min(
                max(int(current_size * 0.5), 1),
                self.max_neurons_per_expansion
            )
            return {
                'type': 'width',
                'layer_idx': layer_idx,
                'num_neurons': num_neurons
            }

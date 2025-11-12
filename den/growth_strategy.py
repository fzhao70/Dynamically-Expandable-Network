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


class BiologicalGrowth(GrowthStrategy):
    """
    Biologically-inspired growth strategy that mimics neural development in living organisms.

    Based on biological principles:
    - Activity-dependent neurogenesis: Active neurons promote growth
    - Hebbian learning: "Neurons that fire together wire together"
    - Synaptic pruning: Weak neurons/connections are pruned
    - Homeostatic plasticity: Maintains balanced network activity
    - Competitive growth: Resources are limited, strongest neurons survive
    - Energy efficiency: Penalizes unnecessarily large networks

    This strategy monitors neuron activity, connection strengths, and network efficiency
    to make growth decisions similar to how brains develop in animals.
    """

    def __init__(
        self,
        patience: int = 12,
        cooldown: int = 8,
        activity_threshold: float = 0.3,
        pruning_threshold: float = 0.1,
        energy_cost_weight: float = 0.01,
        max_neurons_per_expansion: int = 16,
        max_network_size: int = 5000,
        enable_pruning: bool = True,
        hebbian_window: int = 5
    ):
        """
        Args:
            patience: Epochs to wait before considering growth
            cooldown: Epochs to wait after growth/pruning
            activity_threshold: Neuron activity level to trigger growth (0-1)
            pruning_threshold: Activity level below which neurons may be pruned
            energy_cost_weight: Penalty for network size (metabolic cost)
            max_neurons_per_expansion: Maximum neurons to add at once
            max_network_size: Maximum total parameters (resource limit)
            enable_pruning: Whether to prune weak neurons
            hebbian_window: Number of epochs to track activity correlations
        """
        super().__init__(patience, cooldown)
        self.activity_threshold = activity_threshold
        self.pruning_threshold = pruning_threshold
        self.energy_cost_weight = energy_cost_weight
        self.max_neurons_per_expansion = max_neurons_per_expansion
        self.max_network_size = max_network_size
        self.enable_pruning = enable_pruning
        self.hebbian_window = hebbian_window

        # Track neuron activity history (like neurotransmitter levels)
        self.activity_history = []
        # Track layer-wise activity
        self.layer_activity_history = {}
        # Track connection strength changes (synaptic plasticity)
        self.weight_change_history = []
        # Track network efficiency
        self.efficiency_history = []
        # Last pruning epoch
        self.last_pruning_epoch = 0

    def _calculate_neuron_activity(self, network, metrics: Dict[str, float]) -> Dict[int, float]:
        """
        Calculate activity level for each layer (like neural firing rates).

        Activity is based on:
        - Activation magnitudes
        - Weight usage
        - Gradient flow
        """
        layer_activities = {}

        for i, layer in enumerate(network.layers):
            # Get neuron importance as proxy for activity
            importance = layer.get_neuron_importance().cpu().numpy()

            # Normalize to 0-1 range (like firing rate)
            if importance.max() > 0:
                activity = importance / importance.max()
            else:
                activity = np.zeros_like(importance)

            # Calculate average activity for the layer
            layer_activities[i] = float(np.mean(activity))

        return layer_activities

    def _calculate_energy_cost(self, network) -> float:
        """
        Calculate metabolic energy cost of maintaining the network.

        In biological systems, larger brains consume more energy.
        This encourages efficiency.
        """
        num_params = network.get_num_parameters()
        # Nonlinear cost (like actual metabolic cost)
        energy_cost = (num_params ** 1.2) * self.energy_cost_weight
        return energy_cost

    def _calculate_network_efficiency(self, loss: float, network) -> float:
        """
        Calculate network efficiency: performance per parameter.

        Biological systems optimize for both performance and energy efficiency.
        """
        num_params = network.get_num_parameters()
        if num_params == 0 or loss <= 0:
            return 0.0

        # Lower loss and fewer params = higher efficiency
        efficiency = 1.0 / (loss * (1 + num_params / 1000.0))
        return efficiency

    def _detect_plasticity_needs(self, metrics: Dict[str, float]) -> bool:
        """
        Detect if network needs structural plasticity (like neurogenesis in hippocampus).

        Signs:
        - High activity but poor performance (overworked neurons)
        - High gradients (neurons struggling to learn)
        - Declining efficiency
        """
        loss = metrics.get('loss', float('inf'))
        avg_gradient = metrics.get('avg_gradient_norm', 0.0)

        # Track efficiency
        current_efficiency = metrics.get('efficiency', 0.0)
        self.efficiency_history.append(current_efficiency)
        if len(self.efficiency_history) > self.hebbian_window:
            self.efficiency_history.pop(0)

        # Check if network is struggling
        high_gradients = avg_gradient > 0.05
        poor_efficiency = len(self.efficiency_history) >= 3 and \
                         current_efficiency < np.mean(self.efficiency_history) * 0.8

        return high_gradients or poor_efficiency

    def should_grow(
        self,
        metrics: Dict[str, float],
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if network should undergo neurogenesis (create new neurons).

        Based on biological triggers:
        - Persistent high activity with poor performance
        - Learning plateau despite effort
        - Network struggling to form new connections
        """
        current_loss = metrics.get('loss', float('inf'))

        # Check resource limits (like physical brain size limits)
        if network.get_num_parameters() >= self.max_network_size:
            return False, None

        # Calculate energy cost and efficiency
        energy_cost = self._calculate_energy_cost(network)
        efficiency = self._calculate_network_efficiency(current_loss, network)
        metrics['efficiency'] = efficiency

        # Track layer activities
        layer_activities = self._calculate_neuron_activity(network, metrics)
        self.layer_activity_history[epoch] = layer_activities

        # Cooldown period (like refractory period in neurons)
        if self.epochs_since_growth < self.cooldown:
            self.epochs_since_growth += 1
            return False, None

        # Check for improvement
        if current_loss < (self.best_metric - 1e-4):
            self.best_metric = current_loss
            self.epochs_since_improvement = 0
            return False, None

        self.epochs_since_improvement += 1

        # Need sufficient observation period (like brain development stages)
        if self.epochs_since_improvement < self.patience:
            return False, None

        # Detect if structural plasticity is needed
        needs_plasticity = self._detect_plasticity_needs(metrics)

        # Check activity levels (like BDNF signaling in real neurons)
        recent_epochs = [e for e in self.layer_activity_history.keys()
                        if epoch - e < self.hebbian_window]
        if recent_epochs:
            avg_activities = []
            for e in recent_epochs:
                avg_activities.extend(self.layer_activity_history[e].values())
            overall_activity = np.mean(avg_activities)

            # High sustained activity suggests need for growth
            high_activity = overall_activity > self.activity_threshold
        else:
            high_activity = False
            overall_activity = 0.0

        # Growth decision (like adult neurogenesis in hippocampus)
        if (needs_plasticity or high_activity) and self.epochs_since_improvement >= self.patience:
            reason_parts = []
            if needs_plasticity:
                reason_parts.append("network plasticity needed")
            if high_activity:
                reason_parts.append(f"sustained high activity ({overall_activity:.3f})")
            reason_parts.append(f"efficiency={efficiency:.4f}")

            self.epochs_since_improvement = 0
            self.epochs_since_growth = 0

            return True, ", ".join(reason_parts)

        return False, None

    def determine_growth_action(
        self,
        metrics: Dict[str, float],
        network: Any
    ) -> Dict[str, Any]:
        """
        Determine how to grow based on biological principles.

        Mimics:
        - Targeted neurogenesis in active regions
        - Formation of new neural circuits
        - Competitive resource allocation
        """
        layer_sizes = network.get_layer_sizes()
        num_layers = len(layer_sizes)

        # Get recent activity data
        recent_activities = {}
        for layer_idx in range(num_layers):
            activities = []
            for epoch, layer_acts in self.layer_activity_history.items():
                if layer_idx in layer_acts:
                    activities.append(layer_acts[layer_idx])
            if activities:
                recent_activities[layer_idx] = np.mean(activities[-self.hebbian_window:])
            else:
                recent_activities[layer_idx] = 0.0

        # Activity-dependent growth: grow where activity is highest
        # (like neurogenesis in hippocampus during learning)
        if recent_activities:
            most_active_layer = max(recent_activities.items(), key=lambda x: x[1])[0]
            activity_level = recent_activities[most_active_layer]

            # Also consider layer capacity
            layer_capacities = [size for size in layer_sizes]
            smallest_layer = np.argmin(layer_capacities)

            # Weighted decision: favor active layers, but also reinforce small layers
            activity_score = {i: recent_activities.get(i, 0) * 2.0 for i in range(num_layers)}
            capacity_score = {i: 1.0 / (layer_sizes[i] + 1) for i in range(num_layers)}

            combined_score = {
                i: activity_score.get(i, 0) + capacity_score.get(i, 0)
                for i in range(num_layers)
            }

            target_layer = max(combined_score.items(), key=lambda x: x[1])[0]
        else:
            # No activity data, grow smallest layer
            target_layer = np.argmin(layer_sizes)

        # Determine growth magnitude based on activity level
        # (like growth factor concentration in brain)
        current_size = layer_sizes[target_layer]
        activity_factor = recent_activities.get(target_layer, 0.5)

        # More active layers get more neurons (competitive growth)
        base_growth = max(int(current_size * 0.3), 2)
        activity_bonus = int(base_growth * activity_factor)
        num_neurons = min(
            base_growth + activity_bonus,
            self.max_neurons_per_expansion
        )

        # Occasionally add depth for more complex representations
        # (like formation of new brain regions during evolution)
        depth_threshold = 0.7  # High activity threshold for depth growth
        if (num_layers < 6 and
            recent_activities and
            max(recent_activities.values()) > depth_threshold):

            # Add layer in middle (like cortical column formation)
            position = num_layers // 2
            num_neurons_depth = min(
                int(np.mean(layer_sizes)),
                self.max_neurons_per_expansion
            )

            return {
                'type': 'depth',
                'position': position,
                'num_neurons': num_neurons_depth
            }

        # Width expansion (most common)
        return {
            'type': 'width',
            'layer_idx': target_layer,
            'num_neurons': num_neurons
        }

    def should_prune(
        self,
        network: Any,
        epoch: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if synaptic pruning should occur.

        Biological pruning:
        - Removes weak/unused connections
        - Occurs during development and learning
        - "Use it or lose it" principle
        """
        if not self.enable_pruning:
            return False, None

        # Don't prune too frequently (like pruning cycles in development)
        if epoch - self.last_pruning_epoch < self.cooldown * 2:
            return False, None

        # Need sufficient history to identify weak neurons
        if len(self.layer_activity_history) < self.hebbian_window:
            return False, None

        # Check for consistently low-activity neurons
        for layer_idx, layer in enumerate(network.layers):
            importance = layer.get_neuron_importance().cpu().numpy()
            if importance.max() > 0:
                normalized_importance = importance / importance.max()
                weak_neurons = np.sum(normalized_importance < self.pruning_threshold)

                # If >20% of neurons are weak, consider pruning
                if weak_neurons > len(importance) * 0.2 and len(importance) > 8:
                    self.last_pruning_epoch = epoch
                    return True, f"pruning {weak_neurons} weak neurons in layer {layer_idx}"

        return False, None

"""
Core implementation of the Dynamically Expandable Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Callable, Union, Type
import copy

from .layers import ExpandableLinear, ExpandableSequential


class DynamicExpandableNetwork(nn.Module):
    """
    A neural network that can dynamically expand its architecture during training.

    Features:
    - Width expansion: Add neurons to existing layers
    - Depth expansion: Add new layers to the network
    - Automatic growth based on training performance
    - Knowledge preservation during expansion
    - Checkpoint/restart capability
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int],
        activation: Union[nn.Module, Type[nn.Module], Callable[[], nn.Module]] = nn.ReLU,
        output_activation: Optional[Union[nn.Module, Type[nn.Module], Callable[[], nn.Module]]] = None,
        dropout: float = 0.0,
        task_type: str = 'regression'
    ):
        """
        Initialize the Dynamic Expandable Network.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            hidden_sizes: List of hidden layer sizes (initial architecture)
            activation: Activation function (e.g., nn.ReLU, nn.Tanh(), or nn.ReLU())
            output_activation: Activation for output layer (None for regression)
            dropout: Dropout probability (0.0 = no dropout)
            task_type: 'regression' or 'classification'
        """
        super(DynamicExpandableNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.initial_hidden_sizes = hidden_sizes.copy()
        self.hidden_sizes = hidden_sizes.copy()
        self.activation_fn = activation
        self.output_activation_fn = output_activation
        self.dropout_prob = dropout
        self.task_type = task_type

        # Build the network
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0 else None

        # Create hidden layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(ExpandableLinear(prev_size, hidden_size))
            self.activations.append(self._create_activation(activation))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        self.output_layer = ExpandableLinear(prev_size, output_size)
        self.output_activation = self._create_activation(output_activation) if output_activation else None

        # Track network growth
        self.growth_history = []
        self.training_history = []

    def _create_activation(self, activation: Optional[Union[nn.Module, Type[nn.Module], Callable[[], nn.Module]]]) -> Optional[nn.Module]:
        """
        Create an activation module from various input types.

        Args:
            activation: Can be:
                - An nn.Module instance (e.g., nn.ReLU())
                - An nn.Module class (e.g., nn.ReLU)
                - A callable that returns an nn.Module (e.g., lambda: nn.ReLU())
                - None

        Returns:
            An nn.Module instance or None
        """
        if activation is None:
            return None

        # If it's already an instance, return it
        if isinstance(activation, nn.Module):
            return activation

        # If it's a class or callable, instantiate it
        if callable(activation):
            try:
                return activation()
            except TypeError:
                # Might be a class that needs no args
                raise ValueError(f"Activation must be an nn.Module instance, class, or callable that returns an nn.Module")

        raise ValueError(f"Invalid activation type: {type(activation)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
            if self.dropouts is not None and i < len(self.dropouts):
                x = self.dropouts[i](x)

        # Output layer
        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def expand_layer_width(self, layer_idx: int, num_neurons: int):
        """
        Expand a specific layer by adding neurons.

        Args:
            layer_idx: Index of the layer to expand
            num_neurons: Number of neurons to add
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")

        old_size = self.layers[layer_idx].out_features

        # Expand the target layer
        self.layers[layer_idx].expand_width(num_neurons)

        # Expand the input of the next layer
        if layer_idx + 1 < len(self.layers):
            self.layers[layer_idx + 1].expand_input(num_neurons)
        else:
            # This is the last hidden layer, expand output layer input
            self.output_layer.expand_input(num_neurons)

        # Update hidden sizes
        self.hidden_sizes[layer_idx] += num_neurons

        # Record the expansion
        self.growth_history.append({
            'type': 'width',
            'layer_idx': layer_idx,
            'old_size': old_size,
            'new_size': self.layers[layer_idx].out_features,
            'neurons_added': num_neurons
        })

    def expand_depth(self, position: int, num_neurons: int, activation: Optional[Union[nn.Module, Type[nn.Module], Callable[[], nn.Module]]] = None):
        """
        Add a new layer to the network.

        Args:
            position: Position to insert the new layer (0 = after input, len(layers) = before output)
            num_neurons: Number of neurons in the new layer
            activation: Activation function for the new layer (None = use default)
        """
        if position < 0 or position > len(self.layers):
            raise ValueError(f"Invalid position: {position}")

        # Determine input size for new layer
        if position == 0:
            in_features = self.input_size
        else:
            in_features = self.layers[position - 1].out_features

        # Determine output size for new layer
        if position < len(self.layers):
            out_features = num_neurons
            # The next layer will need to expand its input
            next_layer_needs_expansion = True
        else:
            out_features = num_neurons
            next_layer_needs_expansion = False

        # Create new layer
        new_layer = ExpandableLinear(in_features, out_features)

        # Insert the new layer
        self.layers.insert(position, new_layer)

        # Add activation
        if activation is None:
            activation = self.activation_fn
        self.activations.insert(position, self._create_activation(activation))

        # Add dropout if needed
        if self.dropouts is not None:
            self.dropouts.insert(position, nn.Dropout(self.dropout_prob))

        # Expand the next layer's input dimension if needed
        if next_layer_needs_expansion:
            if position + 1 < len(self.layers):
                # Need to create a new layer that combines old and new
                old_next_layer = self.layers[position + 1]

                # Create new layer with expanded input
                new_next_layer = ExpandableLinear(
                    in_features + out_features,
                    old_next_layer.out_features
                )

                # Copy old weights and initialize new connections
                with torch.no_grad():
                    # Copy weights for old inputs
                    new_next_layer.linear.weight[:, :in_features] = old_next_layer.linear.weight
                    # Initialize weights for new inputs (from new layer)
                    nn.init.kaiming_normal_(new_next_layer.linear.weight[:, in_features:])
                    # Copy bias
                    if old_next_layer.use_bias:
                        new_next_layer.linear.bias[:] = old_next_layer.linear.bias

                self.layers[position + 1] = new_next_layer
        else:
            # Adding before output layer
            self.output_layer.expand_input(out_features)

        # Update hidden sizes
        self.hidden_sizes.insert(position, out_features)

        # Record the expansion
        self.growth_history.append({
            'type': 'depth',
            'position': position,
            'num_neurons': num_neurons,
            'total_layers': len(self.layers)
        })

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_sizes(self) -> List[int]:
        """Get current sizes of all layers."""
        return [layer.out_features for layer in self.layers]

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get detailed information about the current architecture."""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': self.get_layer_sizes(),
            'num_layers': len(self.layers),
            'num_parameters': self.get_num_parameters(),
            'activation': str(type(self.activation_fn).__name__) if self.activation_fn else 'None',
            'dropout': self.dropout_prob,
            'task_type': self.task_type,
            'growth_history': self.growth_history
        }

    def reset_to_initial(self):
        """Reset the network to its initial architecture (loses all learned weights)."""
        # Reinitialize
        self.__init__(
            self.input_size,
            self.output_size,
            self.initial_hidden_sizes,
            self.activation_fn,
            self.output_activation_fn,
            self.dropout_prob,
            self.task_type
        )

    def save_checkpoint(self, path: str):
        """
        Save a checkpoint of the current network state.

        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_sizes': self.hidden_sizes,
            'initial_hidden_sizes': self.initial_hidden_sizes,
            'activation_fn': self.activation_fn,
            'output_activation_fn': self.output_activation_fn,
            'dropout': self.dropout_prob,
            'task_type': self.task_type,
            'growth_history': self.growth_history,
            'architecture_info': self.get_architecture_info()
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        """
        Load a network from a checkpoint.

        Args:
            path: Path to the checkpoint
            device: Device to load the model on

        Returns:
            DynamicExpandableNetwork instance
        """
        checkpoint = torch.load(path, map_location=device)

        # Create network with current architecture
        network = cls(
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size'],
            hidden_sizes=checkpoint['initial_hidden_sizes'],
            activation=checkpoint.get('activation_fn', checkpoint.get('activation', nn.ReLU)),  # Backward compat
            output_activation=checkpoint.get('output_activation_fn', checkpoint.get('output_activation')),
            dropout=checkpoint['dropout'],
            task_type=checkpoint['task_type']
        )

        # Replay growth history to restore architecture
        for growth in checkpoint['growth_history']:
            if growth['type'] == 'width':
                network.expand_layer_width(growth['layer_idx'], growth['neurons_added'])
            elif growth['type'] == 'depth':
                network.expand_depth(growth['position'], growth['num_neurons'])

        # Load weights
        network.load_state_dict(checkpoint['state_dict'])

        return network

    def __repr__(self):
        layer_sizes = self.get_layer_sizes()
        return (f"DynamicExpandableNetwork(\n"
                f"  input_size={self.input_size},\n"
                f"  output_size={self.output_size},\n"
                f"  hidden_sizes={layer_sizes},\n"
                f"  num_parameters={self.get_num_parameters()},\n"
                f"  num_expansions={len(self.growth_history)}\n"
                f")")

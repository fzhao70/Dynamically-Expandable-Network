"""
Expandable neural network layers that can grow dynamically during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExpandableLinear(nn.Module):
    """
    A linear layer that can dynamically expand its number of neurons.

    Features:
    - Can add neurons (width expansion)
    - Preserves learned weights when expanding
    - Supports different initialization strategies for new neurons
    """

    def __init__(self, in_features, out_features, bias=True, init_strategy='kaiming'):
        """
        Args:
            in_features: Number of input features
            out_features: Initial number of output features
            bias: Whether to use bias
            init_strategy: Initialization strategy for new neurons ('kaiming', 'xavier', 'zero')
        """
        super(ExpandableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_strategy = init_strategy

        # Create the linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Track expansion history
        self.expansion_history = []

    def forward(self, x):
        """Forward pass through the layer."""
        return self.linear(x)

    def expand_width(self, num_new_neurons):
        """
        Expand the layer by adding new output neurons.

        Args:
            num_new_neurons: Number of neurons to add

        Returns:
            None
        """
        if num_new_neurons <= 0:
            return

        old_out_features = self.out_features
        new_out_features = self.out_features + num_new_neurons

        # Create new linear layer with expanded size
        new_linear = nn.Linear(self.in_features, new_out_features, bias=self.use_bias)

        # Copy old weights
        with torch.no_grad():
            new_linear.weight[:old_out_features] = self.linear.weight
            if self.use_bias:
                new_linear.bias[:old_out_features] = self.linear.bias

            # Initialize new neurons
            if self.init_strategy == 'kaiming':
                nn.init.kaiming_normal_(new_linear.weight[old_out_features:])
            elif self.init_strategy == 'xavier':
                nn.init.xavier_normal_(new_linear.weight[old_out_features:])
            elif self.init_strategy == 'zero':
                new_linear.weight[old_out_features:].zero_()

            if self.use_bias:
                new_linear.bias[old_out_features:].zero_()

        # Replace the old layer
        self.linear = new_linear
        self.out_features = new_out_features

        # Record expansion
        self.expansion_history.append({
            'old_size': old_out_features,
            'new_size': new_out_features,
            'neurons_added': num_new_neurons
        })

    def expand_input(self, num_new_inputs):
        """
        Expand the input dimension of the layer.
        Used when the previous layer expands.

        Args:
            num_new_inputs: Number of input features to add
        """
        if num_new_inputs <= 0:
            return

        old_in_features = self.in_features
        new_in_features = self.in_features + num_new_inputs

        # Create new linear layer with expanded input size
        new_linear = nn.Linear(new_in_features, self.out_features, bias=self.use_bias)

        # Copy old weights
        with torch.no_grad():
            new_linear.weight[:, :old_in_features] = self.linear.weight
            if self.use_bias:
                new_linear.bias[:] = self.linear.bias

            # Initialize weights for new inputs
            if self.init_strategy == 'kaiming':
                nn.init.kaiming_normal_(new_linear.weight[:, old_in_features:])
            elif self.init_strategy == 'xavier':
                nn.init.xavier_normal_(new_linear.weight[:, old_in_features:])
            elif self.init_strategy == 'zero':
                new_linear.weight[:, old_in_features:].zero_()

        # Replace the old layer
        self.linear = new_linear
        self.in_features = new_in_features

    def get_neuron_importance(self):
        """
        Calculate importance scores for each neuron based on weight magnitudes.
        Can be used for pruning or analysis.

        Returns:
            Tensor of importance scores for each output neuron
        """
        with torch.no_grad():
            # L2 norm of weights for each output neuron
            importance = torch.norm(self.linear.weight, p=2, dim=1)
            if self.use_bias:
                importance += torch.abs(self.linear.bias)
            return importance

    def __repr__(self):
        return f"ExpandableLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"


class ExpandableSequential(nn.Sequential):
    """
    A sequential container for expandable layers that can add new layers dynamically.
    """

    def __init__(self, *args):
        super(ExpandableSequential, self).__init__(*args)
        self.depth_expansion_history = []

    def add_layer(self, layer, position=None):
        """
        Add a new layer to the sequence.

        Args:
            layer: The layer to add
            position: Position to insert (None = append at end)
        """
        if position is None:
            position = len(self)

        # Convert to list, insert, and rebuild
        layers = list(self.children())
        layers.insert(position, layer)

        # Clear and rebuild
        for i in range(len(self)):
            delattr(self, str(i))

        for idx, module in enumerate(layers):
            self.add_module(str(idx), module)

        self.depth_expansion_history.append({
            'position': position,
            'total_layers': len(layers)
        })

    def expand_at_position(self, position, num_neurons, activation=None):
        """
        Add a new expandable layer at the specified position.

        Args:
            position: Where to insert the new layer
            num_neurons: Number of neurons in the new layer
            activation: Activation function to add after the layer
        """
        layers = list(self.children())

        # Determine input size from previous layer
        if position > 0 and isinstance(layers[position - 1], ExpandableLinear):
            in_features = layers[position - 1].out_features
        elif position > 0:
            # Try to infer from previous layer
            raise ValueError("Cannot infer input size from previous layer")
        else:
            raise ValueError("Cannot add layer at position 0 without knowing input size")

        # Determine output size from next layer
        if position < len(layers) and isinstance(layers[position], ExpandableLinear):
            # We need to expand the next layer's input
            new_layer = ExpandableLinear(in_features, num_neurons)
            self.add_layer(new_layer, position)

            # Expand next layer's input
            if isinstance(layers[position], ExpandableLinear):
                layers[position].expand_input(num_neurons)
        else:
            new_layer = ExpandableLinear(in_features, num_neurons)
            self.add_layer(new_layer, position)

        # Add activation if specified
        if activation is not None:
            self.add_layer(activation, position + 1)

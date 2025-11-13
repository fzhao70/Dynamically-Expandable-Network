"""
Utility functions for DEN visualization and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None, use_time: bool = False):
    """
    Plot training history including loss and growth events.

    Args:
        history: Training history dictionary from DENTrainer
        save_path: Optional path to save the figure
        use_time: If True, use wall-clock time instead of epochs on x-axis
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Determine x-axis data
    if use_time and 'timestamps' in history and history['timestamps']:
        x_data = [t / 60.0 for t in history['timestamps']]  # Convert to minutes
        x_label = 'Time (minutes)'
    else:
        x_data = history['epochs']
        x_label = 'Epoch'

    # Plot loss
    ax1 = axes[0]
    ax1.plot(x_data, history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
    if history['val_loss']:
        ax1.plot(x_data, history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)

    # Mark growth events
    for event in history['growth_events']:
        if use_time and 'timestamp' in event:
            x_pos = event['timestamp'] / 60.0  # Convert to minutes
        else:
            x_pos = event['epoch']

        ax1.axvline(x=x_pos, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        action = event['action']
        if action['type'] == 'width':
            label = f"W+{action['num_neurons']}"
        else:
            label = f"D+{action['num_neurons']}"
        ax1.text(x_pos, ax1.get_ylim()[1] * 0.95, label, rotation=90, va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress and Network Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot network size evolution
    ax2 = axes[1]
    param_counts = []
    current_params = 0

    # Build parameter counts over time
    if history['growth_events']:
        # Start with initial params
        current_params = history['growth_events'][0]['architecture']['num_parameters']
        # Subtract the first growth to get initial size
        first_growth = history['growth_events'][0]
        if first_growth['action']['type'] == 'width':
            # Rough estimate
            current_params = max(100, current_params - first_growth['action']['num_neurons'] * 50)

    for i in range(len(x_data)):
        # Check if there was a growth event at this point
        if use_time:
            current_time = history['timestamps'][i] if i < len(history['timestamps']) else 0
            for event in history['growth_events']:
                if 'timestamp' in event and abs(event['timestamp'] - current_time) < 0.1:
                    current_params = event['architecture']['num_parameters']
                    break
        else:
            current_epoch = history['epochs'][i]
            for event in history['growth_events']:
                if event['epoch'] == current_epoch:
                    current_params = event['architecture']['num_parameters']
                    break

        param_counts.append(current_params)

    ax2.plot(x_data, param_counts, linewidth=2, color='green', marker='o', markersize=3)

    # Mark growth events on params plot too
    for event in history['growth_events']:
        if use_time and 'timestamp' in event:
            x_pos = event['timestamp'] / 60.0
        else:
            x_pos = event['epoch']
        ax2.axvline(x=x_pos, color='red', linestyle='--', alpha=0.3, linewidth=1)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Network Size Evolution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_architecture_evolution(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Visualize how the network architecture evolved during training.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the figure
    """
    if not history['growth_events']:
        print("No growth events to visualize")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract architecture at each growth event
    architectures = []
    epochs = [0]

    # Add initial architecture (before first growth)
    if history['growth_events']:
        first_event = history['growth_events'][0]
        # Reconstruct initial architecture
        architectures.append([32, 32])  # placeholder

    for event in history['growth_events']:
        epochs.append(event['epoch'])
        arch_info = event['architecture']
        architectures.append(arch_info['hidden_sizes'])

    # Plot architecture evolution
    max_layers = max(len(arch) for arch in architectures)
    colors = plt.cm.viridis(np.linspace(0, 1, max_layers))

    for layer_idx in range(max_layers):
        layer_sizes = []
        layer_epochs = []

        for i, arch in enumerate(architectures):
            if layer_idx < len(arch):
                layer_sizes.append(arch[layer_idx])
                layer_epochs.append(epochs[i])

        if layer_sizes:
            ax.plot(layer_epochs, layer_sizes, marker='o', linewidth=2,
                   label=f'Layer {layer_idx + 1}', color=colors[layer_idx])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Layer Size (neurons)')
    ax.set_title('Layer Size Evolution During Training')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_network_architecture(network, save_path: Optional[str] = None):
    """
    Create a visual representation of the network architecture.

    Args:
        network: DynamicExpandableNetwork instance
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    layer_sizes = [network.input_size] + network.get_layer_sizes() + [network.output_size]
    num_layers = len(layer_sizes)

    # Position layers
    x_positions = np.linspace(0, 10, num_layers)
    max_neurons = max(layer_sizes)

    # Draw nodes
    for i, (x, size) in enumerate(zip(x_positions, layer_sizes)):
        y_positions = np.linspace(0, 10, size + 2)[1:-1]  # Center vertically

        # Sample neurons if too many
        display_size = min(size, 20)
        if size > display_size:
            y_positions = np.linspace(0, 10, display_size + 2)[1:-1]

        for y in y_positions:
            circle = plt.Circle((x, y), 0.15, color='lightblue', ec='darkblue', linewidth=2)
            ax.add_patch(circle)

        # Add ellipsis if truncated
        if size > display_size:
            ax.text(x, 5, '...', ha='center', va='center', fontsize=16, fontweight='bold')

        # Layer labels
        if i == 0:
            label = f'Input\n({size})'
        elif i == num_layers - 1:
            label = f'Output\n({size})'
        else:
            label = f'Hidden {i}\n({size})'

        ax.text(x, -0.5, label, ha='center', va='top', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2, 11)
    ax.axis('off')
    ax.set_title(f'Network Architecture: {network.get_num_parameters()} parameters',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_growth_summary(history: Dict[str, List]):
    """
    Print a summary of growth events during training.

    Args:
        history: Training history dictionary
    """
    print("\n" + "="*70)
    print("NETWORK GROWTH SUMMARY")
    print("="*70)

    if not history['growth_events']:
        print("No growth events occurred during training.")
        return

    print(f"\nTotal growth events: {len(history['growth_events'])}")
    print(f"Training epochs: {len(history['epochs'])}")

    # Count growth types
    width_expansions = sum(1 for e in history['growth_events'] if e['action']['type'] == 'width')
    depth_expansions = sum(1 for e in history['growth_events'] if e['action']['type'] == 'depth')

    print(f"Width expansions: {width_expansions}")
    print(f"Depth expansions: {depth_expansions}")

    # Initial and final architecture
    first_event = history['growth_events'][0]
    last_event = history['growth_events'][-1]

    print(f"\nInitial parameters: {first_event['architecture']['num_parameters']}")
    print(f"Final parameters: {last_event['architecture']['num_parameters']}")
    print(f"Parameter growth: {last_event['architecture']['num_parameters'] / first_event['architecture']['num_parameters']:.2f}x")

    print("\nGrowth Timeline:")
    print("-" * 90)

    # Check if we have timestamps
    has_timestamps = any('timestamp' in e for e in history['growth_events'])

    if has_timestamps:
        print(f"{'Epoch':<8} {'Time (min)':<12} {'Type':<10} {'Details':<30} {'Total Params':<15}")
    else:
        print(f"{'Epoch':<8} {'Type':<10} {'Details':<30} {'Total Params':<15}")
    print("-" * 90)

    for event in history['growth_events']:
        epoch = event['epoch']
        action = event['action']
        params = event['architecture']['num_parameters']

        if action['type'] == 'width':
            details = f"Layer {action['layer_idx']} +{action['num_neurons']} neurons"
        else:
            details = f"New layer at pos {action['position']} ({action['num_neurons']} neurons)"

        if has_timestamps and 'timestamp' in event:
            time_min = event['timestamp'] / 60.0
            print(f"{epoch:<8} {time_min:<12.2f} {action['type'].upper():<10} {details:<30} {params:<15,}")
        else:
            print(f"{epoch:<8} {action['type'].upper():<10} {details:<30} {params:<15,}")

    print("="*90 + "\n")


def analyze_layer_importance(network) -> Dict[int, float]:
    """
    Analyze the importance of each layer based on weight magnitudes.

    Args:
        network: DynamicExpandableNetwork instance

    Returns:
        Dictionary mapping layer index to importance score
    """
    importance_scores = {}

    for i, layer in enumerate(network.layers):
        importance = layer.get_neuron_importance()
        importance_scores[i] = {
            'mean_importance': importance.mean().item(),
            'std_importance': importance.std().item(),
            'max_importance': importance.max().item(),
            'min_importance': importance.min().item()
        }

    return importance_scores


def export_architecture_to_json(network, filepath: str):
    """
    Export network architecture to JSON format.

    Args:
        network: DynamicExpandableNetwork instance
        filepath: Path to save JSON file
    """
    import json

    arch_info = network.get_architecture_info()

    with open(filepath, 'w') as f:
        json.dump(arch_info, f, indent=2)

    print(f"Architecture exported to {filepath}")


def compare_architectures(network1, network2):
    """
    Compare two network architectures.

    Args:
        network1: First network
        network2: Second network
    """
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)

    arch1 = network1.get_architecture_info()
    arch2 = network2.get_architecture_info()

    print(f"\nNetwork 1:")
    print(f"  Layers: {arch1['num_layers']}")
    print(f"  Hidden sizes: {arch1['hidden_sizes']}")
    print(f"  Parameters: {arch1['num_parameters']:,}")
    print(f"  Growth events: {len(arch1['growth_history'])}")

    print(f"\nNetwork 2:")
    print(f"  Layers: {arch2['num_layers']}")
    print(f"  Hidden sizes: {arch2['hidden_sizes']}")
    print(f"  Parameters: {arch2['num_parameters']:,}")
    print(f"  Growth events: {len(arch2['growth_history'])}")

    print(f"\nDifferences:")
    print(f"  Layer difference: {arch2['num_layers'] - arch1['num_layers']}")
    print(f"  Parameter difference: {arch2['num_parameters'] - arch1['num_parameters']:,}")
    print(f"  Parameter ratio: {arch2['num_parameters'] / arch1['num_parameters']:.2f}x")

    print("="*70 + "\n")

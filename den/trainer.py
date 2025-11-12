"""
Training module for Dynamically Expandable Networks with continuous learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any, List, Callable
import time
from pathlib import Path

from .core import DynamicExpandableNetwork
from .growth_strategy import GrowthStrategy, LossBasedGrowth


class DENTrainer:
    """
    Trainer for Dynamic Expandable Networks.

    Features:
    - Continuous learning with automatic network expansion
    - Multiple growth strategies
    - Comprehensive metrics tracking
    - Checkpoint/restart capability
    - Support for both batch and online learning
    """

    def __init__(
        self,
        network: DynamicExpandableNetwork,
        growth_strategy: Optional[GrowthStrategy] = None,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss_function: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        """
        Initialize the DEN Trainer.

        Args:
            network: DynamicExpandableNetwork instance
            growth_strategy: Strategy for network growth (default: LossBasedGrowth)
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            loss_function: Loss function (default: MSE for regression, CrossEntropy for classification)
            device: Device to train on
            verbose: Print training progress
        """
        self.network = network.to(device)
        self.device = device
        self.verbose = verbose

        # Growth strategy
        if growth_strategy is None:
            growth_strategy = LossBasedGrowth()
        self.growth_strategy = growth_strategy

        # Loss function
        if loss_function is None:
            if network.task_type == 'regression':
                self.loss_fn = nn.MSELoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_function

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate)

        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'epochs': [],
            'growth_events': [],
            'architecture_snapshots': []
        }

        # Current epoch
        self.current_epoch = 0

    def _create_optimizer(self, optimizer_name: str, lr: float):
        """Create optimizer instance."""
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.network.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.network.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(self.network.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _recreate_optimizer(self):
        """Recreate optimizer after network growth."""
        self.optimizer = self._create_optimizer(self.optimizer_name, self.learning_rate)

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Dictionary of metrics
        """
        self.network.train()

        total_loss = 0.0
        total_samples = 0
        gradient_norms = []
        layer_gradients = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.network(data)

            # Calculate loss
            if self.network.task_type == 'regression':
                loss = self.loss_fn(output, target)
            else:
                loss = self.loss_fn(output, target.long())

            # Backward pass
            loss.backward()

            # Track gradients
            total_norm = 0.0
            layer_norms = []
            for i, layer in enumerate(self.network.layers):
                if layer.linear.weight.grad is not None:
                    layer_norm = layer.linear.weight.grad.norm().item()
                    layer_norms.append(layer_norm)
                    total_norm += layer_norm ** 2

            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)

            if layer_norms:
                if not layer_gradients:
                    layer_gradients = [[norm] for norm in layer_norms]
                else:
                    for i, norm in enumerate(layer_norms):
                        if i < len(layer_gradients):
                            layer_gradients[i].append(norm)

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        # Calculate metrics
        avg_loss = total_loss / total_samples
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        avg_layer_gradients = [np.mean(grads) for grads in layer_gradients] if layer_gradients else []

        metrics = {
            'loss': avg_loss,
            'avg_gradient_norm': avg_gradient_norm,
            'layer_gradients': avg_layer_gradients
        }

        # Validation
        if val_loader is not None:
            val_loss = self.evaluate(val_loader)
            metrics['val_loss'] = val_loss

        return metrics

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the network on a dataset.

        Args:
            data_loader: Data loader for evaluation

        Returns:
            Average loss
        """
        self.network.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.network(data)

                if self.network.task_type == 'regression':
                    loss = self.loss_fn(output, target)
                else:
                    loss = self.loss_fn(output, target.long())

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        return total_loss / total_samples

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        enable_growth: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 10
    ) -> Dict[str, List]:
        """
        Train the network with automatic growth.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            enable_growth: Whether to enable automatic growth
            checkpoint_dir: Directory to save checkpoints
            checkpoint_frequency: Save checkpoint every N epochs

        Returns:
            Training history dictionary
        """
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch += 1

            # Train one epoch
            metrics = self.train_epoch(train_loader, val_loader)

            # Update history
            self.history['loss'].append(metrics['loss'])
            self.history['epochs'].append(self.current_epoch)
            if 'val_loss' in metrics:
                self.history['val_loss'].append(metrics['val_loss'])

            # Print progress
            if self.verbose:
                log_str = f"Epoch {self.current_epoch}/{self.current_epoch - 1 + epochs}: "
                log_str += f"Loss={metrics['loss']:.6f}"
                if 'val_loss' in metrics:
                    log_str += f", Val Loss={metrics['val_loss']:.6f}"
                log_str += f", Params={self.network.get_num_parameters()}"
                print(log_str)

            # Check for growth
            if enable_growth:
                should_grow, reason = self.growth_strategy.should_grow(
                    metrics,
                    self.network,
                    self.current_epoch
                )

                if should_grow:
                    if self.verbose:
                        print(f"\n🌱 Growth triggered: {reason}")

                    # Determine growth action
                    growth_action = self.growth_strategy.determine_growth_action(
                        metrics,
                        self.network
                    )

                    # Perform growth
                    if growth_action['type'] == 'width':
                        self.network.expand_layer_width(
                            growth_action['layer_idx'],
                            growth_action['num_neurons']
                        )
                        if self.verbose:
                            print(f"   Expanded layer {growth_action['layer_idx']} "
                                  f"by {growth_action['num_neurons']} neurons")

                    elif growth_action['type'] == 'depth':
                        self.network.expand_depth(
                            growth_action['position'],
                            growth_action['num_neurons']
                        )
                        if self.verbose:
                            print(f"   Added layer at position {growth_action['position']} "
                                  f"with {growth_action['num_neurons']} neurons")

                    # Record growth event
                    self.history['growth_events'].append({
                        'epoch': self.current_epoch,
                        'action': growth_action,
                        'reason': reason,
                        'architecture': self.network.get_architecture_info()
                    })

                    # Recreate optimizer with new parameters
                    self._recreate_optimizer()

                    if self.verbose:
                        print(f"   New architecture: {self.network.get_layer_sizes()}")
                        print(f"   Total parameters: {self.network.get_num_parameters()}\n")

            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_file = checkpoint_path / f"checkpoint_epoch_{self.current_epoch}.pt"
                self.save_checkpoint(str(checkpoint_file))

        # Training complete
        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\n✓ Training complete in {elapsed_time:.2f}s")
            print(f"  Final loss: {self.history['loss'][-1]:.6f}")
            print(f"  Architecture: {self.network.get_layer_sizes()}")
            print(f"  Total parameters: {self.network.get_num_parameters()}")
            print(f"  Growth events: {len(self.history['growth_events'])}")

        return self.history

    def continual_learning(
        self,
        new_data: torch.Tensor,
        new_targets: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Continue learning on new data.

        Args:
            new_data: New input data
            new_targets: New target data
            epochs: Number of epochs to train
            batch_size: Batch size

        Returns:
            Final metrics
        """
        # Create data loader
        dataset = TensorDataset(new_data, new_targets)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train on new data
        if self.verbose:
            print(f"\n📚 Continual learning on {len(new_data)} new samples...")

        history = self.train(
            train_loader=data_loader,
            epochs=epochs,
            enable_growth=True
        )

        return {
            'final_loss': history['loss'][-1],
            'growth_events': len(history['growth_events'])
        }

    def save_checkpoint(self, path: str):
        """
        Save a complete training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'network_config': self.network.get_architecture_info(),
            'training_history': self.history,
            'current_epoch': self.current_epoch,
            'growth_strategy': {
                'type': type(self.growth_strategy).__name__,
                'state': self.growth_strategy.__dict__
            }
        }
        torch.save(checkpoint, path)

        if self.verbose:
            print(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load a training checkpoint and restart.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load network state
        self.network.load_state_dict(checkpoint['network_state'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load training history
        self.history = checkpoint['training_history']
        self.current_epoch = checkpoint['current_epoch']

        if self.verbose:
            print(f"📂 Checkpoint loaded from {path}")
            print(f"   Resuming from epoch {self.current_epoch}")
            print(f"   Architecture: {self.network.get_layer_sizes()}")

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on new data.

        Args:
            data: Input data

        Returns:
            Predictions
        """
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            predictions = self.network(data)
        return predictions.cpu()

"""
Simple regression example with Dynamic Expandable Network.

This example demonstrates:
1. Creating a DEN for regression
2. Training with automatic growth
3. Visualizing results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth
from den.utils import plot_training_history, print_growth_summary, visualize_network_architecture


def generate_nonlinear_data(n_samples=1000, noise=0.1):
    """
    Generate non-linear regression data.
    A complex function that may require network growth to learn well.
    """
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (n_samples, 3))

    # Complex non-linear function
    y = (np.sin(X[:, 0] * 2) +
         0.5 * X[:, 1]**2 +
         np.cos(X[:, 2]) * X[:, 0] +
         0.3 * X[:, 0] * X[:, 1])

    y = y + np.random.normal(0, noise, n_samples)

    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


def main():
    print("="*70)
    print("Dynamic Expandable Network - Simple Regression Example")
    print("="*70)

    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_nonlinear_data(n_samples=800)
    X_val, y_val = generate_nonlinear_data(n_samples=200)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create network with small initial size
    print("\n2. Creating Dynamic Expandable Network...")
    network = DynamicExpandableNetwork(
        input_size=3,
        output_size=1,
        hidden_sizes=[8, 8],  # Start small!
        activation=nn.ReLU,  # Pass nn.Module class directly
        task_type='regression'
    )

    print(f"   Initial architecture: {network.get_layer_sizes()}")
    print(f"   Initial parameters: {network.get_num_parameters()}")

    # Create growth strategy
    growth_strategy = LossBasedGrowth(
        patience=15,  # Wait 15 epochs before growing
        cooldown=5,   # Wait 5 epochs after growth
        min_delta=1e-4,
        width_growth_ratio=0.5,
        max_neurons_per_expansion=16
    )

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = DENTrainer(
        network=network,
        growth_strategy=growth_strategy,
        optimizer=torch.optim.Adam,  # Pass optimizer class directly
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )

    # Train the network
    print("\n4. Training network with automatic growth...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        enable_growth=True
    )

    # Print growth summary
    print_growth_summary(history)

    # Final architecture
    print("\n5. Final network architecture:")
    print(f"   Hidden layers: {network.get_layer_sizes()}")
    print(f"   Total parameters: {network.get_num_parameters()}")
    print(f"   Growth events: {len(history['growth_events'])}")

    # Evaluate final performance
    print("\n6. Evaluating final performance...")
    final_train_loss = trainer.evaluate(train_loader)
    final_val_loss = trainer.evaluate(val_loader)
    print(f"   Final training loss: {final_train_loss:.6f}")
    print(f"   Final validation loss: {final_val_loss:.6f}")

    # Make predictions
    print("\n7. Making predictions on test data...")
    X_test, y_test = generate_nonlinear_data(n_samples=100)
    X_test_tensor = torch.FloatTensor(X_test)
    predictions = trainer.predict(X_test_tensor)

    # Calculate R² score
    ss_res = np.sum((y_test - predictions.numpy()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    print(f"   R² Score: {r2_score:.4f}")

    # Save checkpoint
    print("\n8. Saving checkpoint...")
    network.save_checkpoint('checkpoint_final.pt')

    # Visualizations
    print("\n9. Creating visualizations...")
    try:
        plot_training_history(history, save_path='training_history_epochs.png')
        plot_training_history(history, save_path='training_history_time.png', use_time=True)
        visualize_network_architecture(network, save_path='final_architecture.png')
        print("   ✓ Visualizations saved (both epoch-based and time-based)")
    except Exception as e:
        print(f"   Warning: Could not create visualizations: {e}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()

"""
Continual Learning Example with Dynamic Expandable Network.

This example demonstrates:
1. Training on initial data
2. Continually learning from new data streams
3. Network automatically growing to accommodate new patterns
4. Checkpoint/restart capability
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.append('..')

from den import DynamicExpandableNetwork, DENTrainer, AdaptiveGrowth
from den.utils import print_growth_summary


def generate_task_data(task_id, n_samples=500):
    """
    Generate data for different tasks.
    Each task has a different underlying pattern.
    """
    np.random.seed(task_id * 100)
    X = np.random.uniform(-2, 2, (n_samples, 5))

    if task_id == 0:
        # Linear task
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    elif task_id == 1:
        # Quadratic task
        y = X[:, 0]**2 + X[:, 1]**2 - 0.5 * X[:, 2]
    elif task_id == 2:
        # Trigonometric task
        y = np.sin(X[:, 0] * np.pi) + np.cos(X[:, 1] * np.pi)
    elif task_id == 3:
        # Mixed complex task
        y = X[:, 0]**3 + np.sin(X[:, 1]) * X[:, 2] + np.exp(X[:, 3] * 0.5)
    else:
        # Combined patterns
        y = (X[:, 0] * X[:, 1] +
             np.sin(X[:, 2]) +
             X[:, 3]**2 +
             np.cos(X[:, 4]))

    y = y + np.random.normal(0, 0.1, n_samples)
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


def main():
    print("="*70)
    print("Dynamic Expandable Network - Continual Learning Example")
    print("="*70)

    # Initial setup
    print("\n1. Creating initial network...")
    network = DynamicExpandableNetwork(
        input_size=5,
        output_size=1,
        hidden_sizes=[16, 16],
        activation=nn.ReLU,
        task_type='regression'
    )

    print(f"   Initial architecture: {network.get_layer_sizes()}")
    print(f"   Initial parameters: {network.get_num_parameters()}")

    # Use adaptive growth strategy
    growth_strategy = AdaptiveGrowth(
        patience=10,
        cooldown=5,
        loss_threshold=1e-4,
        max_neurons_per_expansion=32
    )

    # Create trainer
    trainer = DENTrainer(
        network=network,
        growth_strategy=growth_strategy,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        verbose=True
    )

    # Number of tasks to learn sequentially
    num_tasks = 5

    all_test_data = []
    all_test_targets = []

    print("\n2. Sequential Task Learning")
    print("-" * 70)

    for task_id in range(num_tasks):
        print(f"\n📚 Learning Task {task_id + 1}/{num_tasks}")
        print("-" * 70)

        # Generate task data
        X_train, y_train = generate_task_data(task_id, n_samples=400)
        X_val, y_val = generate_task_data(task_id, n_samples=100)

        # Store test data for later evaluation
        X_test, y_test = generate_task_data(task_id, n_samples=100)
        all_test_data.append((X_test, y_test, task_id))

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Train on this task
        print(f"\nTraining on task {task_id + 1}...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=30,
            enable_growth=True
        )

        print(f"\n✓ Task {task_id + 1} completed")
        print(f"   Final loss: {history['loss'][-1]:.6f}")
        print(f"   Architecture: {network.get_layer_sizes()}")
        print(f"   Parameters: {network.get_num_parameters()}")
        print(f"   Growth events in this task: {len([e for e in history['growth_events'] if e['epoch'] > task_id * 30])}")

        # Save checkpoint after each task
        checkpoint_path = f'checkpoint_task_{task_id + 1}.pt'
        trainer.save_checkpoint(checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")

    # Final evaluation on all tasks
    print("\n" + "="*70)
    print("3. Evaluating on All Tasks (Testing Catastrophic Forgetting)")
    print("="*70)

    all_losses = []

    for X_test, y_test, task_id in all_test_data:
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        predictions = trainer.predict(X_test_tensor)

        # Calculate MSE
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        all_losses.append(mse)

        print(f"   Task {task_id + 1} - MSE: {mse:.6f}")

    avg_loss = np.mean(all_losses)
    print(f"\n   Average MSE across all tasks: {avg_loss:.6f}")

    # Print final summary
    print("\n" + "="*70)
    print("4. Final Network Summary")
    print("="*70)
    print(f"   Initial architecture: [16, 16]")
    print(f"   Final architecture: {network.get_layer_sizes()}")
    print(f"   Initial parameters: ~400")
    print(f"   Final parameters: {network.get_num_parameters()}")
    print(f"   Growth factor: {network.get_num_parameters() / 400:.2f}x")
    print(f"   Total tasks learned: {num_tasks}")

    # Print complete growth history
    print_growth_summary(trainer.history)

    # Demonstrate restart capability
    print("\n5. Demonstrating Restart Capability")
    print("-" * 70)
    print("   Loading checkpoint from Task 3...")

    # Create new network and trainer
    new_network = DynamicExpandableNetwork(
        input_size=5,
        output_size=1,
        hidden_sizes=[16, 16],
        activation=nn.ReLU,
        task_type='regression'
    )

    new_trainer = DENTrainer(
        network=new_network,
        growth_strategy=AdaptiveGrowth(),
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        verbose=True
    )

    # Load checkpoint
    try:
        new_trainer.load_checkpoint('checkpoint_task_3.pt')
        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   Restarted architecture: {new_network.get_layer_sizes()}")
        print(f"   Restarted parameters: {new_network.get_num_parameters()}")

        # Could continue training from here
        print("\n   Network can now continue learning from this checkpoint!")
    except Exception as e:
        print(f"   Could not load checkpoint: {e}")

    print("\n" + "="*70)
    print("Continual Learning Example Completed!")
    print("="*70)
    print("\nKey Observations:")
    print("  • Network automatically grew to accommodate new tasks")
    print("  • Can save and restart from any checkpoint")
    print("  • Architecture adapts based on task complexity")
    print("="*70)


if __name__ == '__main__':
    main()

"""
Classification Example with Dynamic Expandable Network.

This example demonstrates:
1. Using DEN for classification tasks
2. Training on MNIST-like synthetic data
3. Monitoring classification accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('..')

from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth
from den.utils import print_growth_summary


def generate_classification_data(n_samples=2000, n_features=20, n_classes=5):
    """
    Generate synthetic classification data.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X.astype(np.float32), y.astype(np.int64)


def calculate_accuracy(network, data_loader, device):
    """Calculate classification accuracy."""
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100 * correct / total


def main():
    print("="*70)
    print("Dynamic Expandable Network - Classification Example")
    print("="*70)

    # Generate data
    print("\n1. Generating classification data...")
    X, y = generate_classification_data(n_samples=2000, n_features=20, n_classes=5)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Number of classes: 5")
    print(f"   Number of features: 20")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create network
    print("\n2. Creating Dynamic Expandable Network for classification...")
    network = DynamicExpandableNetwork(
        input_size=20,
        output_size=5,  # 5 classes
        hidden_sizes=[16, 16],  # Start small
        activation=nn.ReLU,
        task_type='classification'
    )

    print(f"   Initial architecture: {network.get_layer_sizes()}")
    print(f"   Initial parameters: {network.get_num_parameters()}")

    # Create growth strategy
    growth_strategy = LossBasedGrowth(
        patience=15,
        cooldown=5,
        min_delta=1e-4,
        width_growth_ratio=0.5,
        max_neurons_per_expansion=24
    )

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n3. Creating trainer (device: {device})...")

    trainer = DENTrainer(
        network=network,
        growth_strategy=growth_strategy,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        loss_function=nn.CrossEntropyLoss(),
        device=device,
        verbose=True
    )

    # Calculate initial accuracy
    initial_train_acc = calculate_accuracy(network, train_loader, device)
    initial_val_acc = calculate_accuracy(network, val_loader, device)
    print(f"   Initial training accuracy: {initial_train_acc:.2f}%")
    print(f"   Initial validation accuracy: {initial_val_acc:.2f}%")

    # Train the network
    print("\n4. Training network with automatic growth...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=80,
        enable_growth=True
    )

    # Print growth summary
    print_growth_summary(history)

    # Final architecture
    print("\n5. Final network architecture:")
    print(f"   Hidden layers: {network.get_layer_sizes()}")
    print(f"   Total parameters: {network.get_num_parameters()}")
    print(f"   Growth events: {len(history['growth_events'])}")

    # Calculate final accuracies
    print("\n6. Final Performance:")
    print("-" * 70)

    train_acc = calculate_accuracy(network, train_loader, device)
    val_acc = calculate_accuracy(network, val_loader, device)
    test_acc = calculate_accuracy(network, test_loader, device)

    print(f"   Training accuracy:   {train_acc:.2f}%")
    print(f"   Validation accuracy: {val_acc:.2f}%")
    print(f"   Test accuracy:       {test_acc:.2f}%")

    # Improvement
    print(f"\n   Improvement from initial:")
    print(f"   Training:   {train_acc - initial_train_acc:+.2f}%")
    print(f"   Validation: {val_acc - initial_val_acc:+.2f}%")

    # Analyze predictions
    print("\n7. Analyzing predictions...")
    network.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Per-class accuracy
    print("\n   Per-class accuracy:")
    for class_idx in range(5):
        mask = all_targets == class_idx
        class_acc = 100 * (all_predictions[mask] == all_targets[mask]).sum() / mask.sum()
        print(f"   Class {class_idx}: {class_acc:.2f}%")

    # Save final model
    print("\n8. Saving final model...")
    network.save_checkpoint('classification_model_final.pt')
    print("   ✓ Model saved to 'classification_model_final.pt'")

    print("\n" + "="*70)
    print("Classification Example Completed Successfully!")
    print("="*70)


if __name__ == '__main__':
    main()

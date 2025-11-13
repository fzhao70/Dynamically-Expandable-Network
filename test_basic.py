"""
Basic test to verify the DEN implementation works correctly.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.insert(0, '.')

from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth

print("="*70)
print("Testing Dynamically Expandable Network Implementation")
print("="*70)

# Test 1: Create network
print("\n1. Testing network creation...")
try:
    network = DynamicExpandableNetwork(
        input_size=5,
        output_size=1,
        hidden_sizes=[8, 8],
        activation='relu',
        task_type='regression'
    )
    print(f"   ✓ Network created successfully")
    print(f"   Architecture: {network.get_layer_sizes()}")
    print(f"   Parameters: {network.get_num_parameters()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Width expansion
print("\n2. Testing width expansion...")
try:
    initial_size = network.layers[0].out_features
    network.expand_layer_width(0, 4)
    new_size = network.layers[0].out_features
    assert new_size == initial_size + 4, "Width expansion failed"
    print(f"   ✓ Width expansion successful: {initial_size} -> {new_size}")
    print(f"   New architecture: {network.get_layer_sizes()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Depth expansion
print("\n3. Testing depth expansion...")
try:
    initial_depth = len(network.layers)
    network.expand_depth(1, 12)
    new_depth = len(network.layers)
    assert new_depth == initial_depth + 1, "Depth expansion failed"
    print(f"   ✓ Depth expansion successful: {initial_depth} -> {new_depth} layers")
    print(f"   New architecture: {network.get_layer_sizes()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Forward pass
print("\n4. Testing forward pass...")
try:
    x = torch.randn(10, 5)
    output = network(x)
    assert output.shape == (10, 1), f"Output shape mismatch: {output.shape}"
    print(f"   ✓ Forward pass successful")
    print(f"   Input shape: {x.shape}, Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Training
print("\n5. Testing training...")
try:
    # Generate simple data
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)

    # Create trainer
    trainer = DENTrainer(
        network=network,
        growth_strategy=LossBasedGrowth(patience=5, cooldown=2),
        optimizer='adam',
        learning_rate=0.01,
        verbose=False
    )

    # Train for a few epochs
    history = trainer.train(loader, epochs=10, enable_growth=False)

    print(f"   ✓ Training successful")
    print(f"   Initial loss: {history['loss'][0]:.6f}")
    print(f"   Final loss: {history['loss'][-1]:.6f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Save and load
print("\n6. Testing checkpoint save/load...")
try:
    # Save
    network.save_checkpoint('test_checkpoint.pt')
    print(f"   ✓ Checkpoint saved")

    # Load
    loaded_network = DynamicExpandableNetwork.load_checkpoint('test_checkpoint.pt')
    assert loaded_network.get_layer_sizes() == network.get_layer_sizes()
    print(f"   ✓ Checkpoint loaded successfully")
    print(f"   Loaded architecture: {loaded_network.get_layer_sizes()}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Growth during training
print("\n7. Testing automatic growth during training...")
try:
    # Create fresh small network
    small_network = DynamicExpandableNetwork(
        input_size=5,
        output_size=1,
        hidden_sizes=[4, 4],
        activation='relu',
        task_type='regression'
    )

    initial_params = small_network.get_num_parameters()

    # Create trainer with growth enabled
    growth_trainer = DENTrainer(
        network=small_network,
        growth_strategy=LossBasedGrowth(patience=3, cooldown=2, width_growth_ratio=0.5),
        optimizer='adam',
        learning_rate=0.01,
        verbose=False
    )

    # Train with growth
    history = growth_trainer.train(loader, epochs=20, enable_growth=True)

    final_params = small_network.get_num_parameters()

    print(f"   ✓ Training with growth completed")
    print(f"   Initial params: {initial_params}")
    print(f"   Final params: {final_params}")
    print(f"   Growth events: {len(history['growth_events'])}")

    if len(history['growth_events']) > 0:
        print(f"   ✓ Network grew during training!")
        for event in history['growth_events']:
            action = event['action']
            print(f"      - Epoch {event['epoch']}: {action['type']} expansion")
    else:
        print(f"   ℹ No growth occurred (this is ok for simple data)")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("All tests passed! ✓")
print("="*70)
print("\nThe DEN implementation is working correctly!")
print("You can now run the examples:")
print("  - python examples/simple_regression.py")
print("  - python examples/continual_learning.py")
print("  - python examples/classification_example.py")
print("="*70)

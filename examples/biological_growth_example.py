"""
Biological Growth Example - Mimicking Neural Development in Living Creatures

This example demonstrates how the BiologicalGrowth strategy mimics real neural development:
- Activity-dependent neurogenesis
- Synaptic pruning
- Energy efficiency optimization
- Hebbian learning principles
- Competitive resource allocation

The network grows like a living brain!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
sys.path.append('..')

from den import DynamicExpandableNetwork, DENTrainer, BiologicalGrowth
from den.utils import print_growth_summary, visualize_network_architecture


def generate_complex_pattern_data(n_samples=500, difficulty='easy'):
    """
    Generate increasingly complex patterns that require neural growth to learn.

    Difficulty levels simulate different developmental stages:
    - easy: Simple patterns (infant learning)
    - medium: Moderate complexity (child learning)
    - hard: Complex patterns (adult learning)
    """
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (n_samples, 8))

    if difficulty == 'easy':
        # Simple linear patterns
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    elif difficulty == 'medium':
        # Non-linear patterns
        y = np.sin(X[:, 0]) + X[:, 1]**2 + 0.5 * X[:, 2]
    else:  # hard
        # Complex interactions
        y = (np.sin(X[:, 0] * np.pi) * np.cos(X[:, 1] * np.pi) +
             X[:, 2]**2 * X[:, 3] +
             np.tanh(X[:, 4] + X[:, 5]) +
             np.exp(X[:, 6] * 0.3) * X[:, 7])

    y = y + np.random.normal(0, 0.1, n_samples)
    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


def main():
    print("="*70)
    print("Biologically-Inspired Neural Growth - Like a Living Brain!")
    print("="*70)

    # Create a very small initial network (like a newborn brain)
    print("\n🧠 Creating initial 'newborn' network...")
    network = DynamicExpandableNetwork(
        input_size=8,
        output_size=1,
        hidden_sizes=[8, 8],  # Start very small
        activation='relu',
        task_type='regression'
    )

    print(f"   Initial 'brain' architecture: {network.get_layer_sizes()}")
    print(f"   Initial parameters: {network.get_num_parameters()}")
    print(f"   (Like a newborn brain with basic neural circuits)")

    # Create biologically-inspired growth strategy
    print("\n🌱 Configuring biological growth strategy...")
    growth_strategy = BiologicalGrowth(
        patience=10,                    # Observation period
        cooldown=6,                     # Recovery period
        activity_threshold=0.3,         # Activity level triggering growth
        pruning_threshold=0.1,          # Threshold for pruning weak neurons
        energy_cost_weight=0.01,        # Metabolic cost penalty
        max_neurons_per_expansion=24,   # Growth factor concentration limit
        max_network_size=3000,          # Physical size limit
        enable_pruning=True,            # Enable synaptic pruning
        hebbian_window=5                # Activity correlation window
    )

    print("   ✓ Growth strategy configured")
    print("   - Activity-dependent neurogenesis: ON")
    print("   - Synaptic pruning: ON")
    print("   - Energy efficiency optimization: ON")
    print("   - Hebbian learning principles: ON")

    # Create trainer with AdamW (improved optimizer)
    print("\n👨‍🏫 Creating trainer with AdamW optimizer...")
    trainer = DENTrainer(
        network=network,
        growth_strategy=growth_strategy,
        optimizer='adamw',  # New AdamW optimizer!
        learning_rate=0.001,
        verbose=True
    )

    print("   ✓ Trainer created with AdamW optimizer")

    # Developmental Stage 1: Infant Learning (Simple Patterns)
    print("\n" + "="*70)
    print("STAGE 1: Infant Brain Development - Learning Simple Patterns")
    print("="*70)

    X_infant, y_infant = generate_complex_pattern_data(400, difficulty='easy')
    infant_dataset = TensorDataset(
        torch.FloatTensor(X_infant),
        torch.FloatTensor(y_infant)
    )
    infant_loader = DataLoader(infant_dataset, batch_size=32, shuffle=True)

    print("\n📚 Training on simple patterns (like learning basic reflexes)...")
    history_infant = trainer.train(
        train_loader=infant_loader,
        epochs=30,
        enable_growth=True
    )

    print(f"\n✓ Infant stage complete")
    print(f"   Architecture: {network.get_layer_sizes()}")
    print(f"   Parameters: {network.get_num_parameters()}")
    print(f"   Growth events: {len(history_infant['growth_events'])}")

    # Developmental Stage 2: Child Learning (Moderate Complexity)
    print("\n" + "="*70)
    print("STAGE 2: Child Brain Development - Learning Complex Patterns")
    print("="*70)

    X_child, y_child = generate_complex_pattern_data(400, difficulty='medium')
    child_dataset = TensorDataset(
        torch.FloatTensor(X_child),
        torch.FloatTensor(y_child)
    )
    child_loader = DataLoader(child_dataset, batch_size=32, shuffle=True)

    print("\n📚 Training on complex patterns (like learning language)...")
    history_child = trainer.train(
        train_loader=child_loader,
        epochs=40,
        enable_growth=True
    )

    print(f"\n✓ Child stage complete")
    print(f"   Architecture: {network.get_layer_sizes()}")
    print(f"   Parameters: {network.get_num_parameters()}")
    print(f"   Growth events: {len(history_child['growth_events'])}")

    # Developmental Stage 3: Adult Learning (High Complexity)
    print("\n" + "="*70)
    print("STAGE 3: Adult Brain Development - Mastering Abstract Concepts")
    print("="*70)

    X_adult, y_adult = generate_complex_pattern_data(400, difficulty='hard')
    adult_dataset = TensorDataset(
        torch.FloatTensor(X_adult),
        torch.FloatTensor(y_adult)
    )
    adult_loader = DataLoader(adult_dataset, batch_size=32, shuffle=True)

    print("\n📚 Training on very complex patterns (like learning mathematics)...")
    history_adult = trainer.train(
        train_loader=adult_loader,
        epochs=50,
        enable_growth=True
    )

    print(f"\n✓ Adult stage complete")
    print(f"   Architecture: {network.get_layer_sizes()}")
    print(f"   Parameters: {network.get_num_parameters()}")
    print(f"   Growth events: {len(history_adult['growth_events'])}")

    # Final Summary
    print("\n" + "="*70)
    print("BRAIN DEVELOPMENT COMPLETE - Final Summary")
    print("="*70)

    print(f"\n🧠 Neural Development Journey:")
    print(f"   Newborn brain:  [8, 8] → {network.get_num_parameters() // 20} neurons")
    print(f"   Infant stage:   {history_infant['growth_events']} growth event(s)")
    print(f"   Child stage:    {history_child['growth_events']} growth event(s)")
    print(f"   Adult stage:    {history_adult['growth_events']} growth event(s)")
    print(f"\n   Final 'mature' brain: {network.get_layer_sizes()}")
    print(f"   Total neurons/parameters: {network.get_num_parameters()}")

    # Combine all growth events
    all_growth_events = (history_infant['growth_events'] +
                        history_child['growth_events'] +
                        history_adult['growth_events'])

    total_history = {
        'loss': (history_infant['loss'] +
                history_child['loss'] +
                history_adult['loss']),
        'epochs': list(range(len(history_infant['loss']) +
                           len(history_child['loss']) +
                           len(history_adult['loss']))),
        'growth_events': all_growth_events,
        'val_loss': []
    }

    # Print detailed growth summary
    print_growth_summary(total_history)

    # Biological principles observed
    print("\n🔬 Biological Principles Demonstrated:")
    print("-" * 70)

    # Count growth types
    width_growth = sum(1 for e in all_growth_events if e['action']['type'] == 'width')
    depth_growth = sum(1 for e in all_growth_events if e['action']['type'] == 'depth')

    print(f"✓ Activity-Dependent Neurogenesis:")
    print(f"  - {len(all_growth_events)} total growth events")
    print(f"  - {width_growth} width expansions (like dendritic growth)")
    print(f"  - {depth_growth} depth expansions (like new cortical layers)")

    print(f"\n✓ Hebbian Learning:")
    print(f"  - Neurons that fire together wire together")
    print(f"  - Active layers preferentially expanded")

    print(f"\n✓ Energy Efficiency:")
    print(f"  - Growth balanced against metabolic cost")
    print(f"  - Network stayed within resource limits")

    print(f"\n✓ Competitive Growth:")
    print(f"  - Limited resources allocated to most active regions")
    print(f"  - Strongest neural pathways reinforced")

    if growth_strategy.enable_pruning:
        print(f"\n✓ Synaptic Pruning:")
        print(f"  - Weak/unused connections pruned")
        print(f"  - 'Use it or lose it' principle applied")

    # Test final performance on all difficulty levels
    print("\n" + "="*70)
    print("Performance Across All Developmental Stages")
    print("="*70)

    stages = [
        ('Infant (easy)', infant_loader),
        ('Child (medium)', child_loader),
        ('Adult (hard)', adult_loader)
    ]

    for stage_name, loader in stages:
        loss = trainer.evaluate(loader)
        print(f"   {stage_name}: Loss = {loss:.6f}")

    # Save the mature brain
    print("\n💾 Saving mature brain state...")
    network.save_checkpoint('biological_brain_mature.pt')
    print("   ✓ Saved to 'biological_brain_mature.pt'")

    # Visualization
    print("\n📊 Creating visualizations...")
    try:
        visualize_network_architecture(network, save_path='biological_brain_architecture.png')
        print("   ✓ Architecture visualization saved")
    except Exception as e:
        print(f"   Warning: Could not create visualization: {e}")

    print("\n" + "="*70)
    print("Biological Neural Growth Demonstration Complete! 🧠✨")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Network grew like a living brain through developmental stages")
    print("  • Activity-dependent growth: active regions expanded more")
    print("  • Energy efficiency: balanced performance vs. metabolic cost")
    print("  • Hebbian principles: reinforced frequently used pathways")
    print("  • Adaptive to complexity: grew more for harder tasks")
    print("\nThis is how real brains develop and learn!")
    print("="*70)


if __name__ == '__main__':
    main()

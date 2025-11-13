# Dynamically Expandable Network (DEN)

A PyTorch implementation of **Dynamically Expandable Neural Networks** that can automatically grow their architecture during training to accommodate complex patterns and new data.

## 🌟 Features

- **Automatic Growth**: Networks expand automatically based on training dynamics
- **Width Expansion**: Add neurons to existing layers when needed
- **Depth Expansion**: Add new layers to increase network capacity
- **Continual Learning**: Learn from new data streams without forgetting
- **Multiple Growth Strategies**: Choose from loss-based, gradient-based, adaptive, or **biological growth**
- **Biological Growth** 🧠 **NEW!**: Mimics real neural development in living creatures
  - Activity-dependent neurogenesis
  - Hebbian learning principles
  - Synaptic pruning of weak neurons
  - Energy efficiency optimization
- **AdamW Optimizer**: Improved optimizer with weight decay
- **Checkpoint/Restart**: Save and resume training at any point
- **PyTorch Native**: Built entirely with PyTorch for seamless integration

## 📦 Installation

### From Source

```bash
git clone https://github.com/yourusername/Dynamically-Expandable-Network.git
cd Dynamically-Expandable-Network
pip install -e .
```

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.0.0

## 🚀 Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth
from den.utils import plot_training_history

# Create your data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# Create a DEN with initial small architecture
network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[16, 16],  # Start small!
    activation=nn.ReLU,     # Pass nn.Module class directly
    task_type='regression'
)

# Set up growth strategy
growth_strategy = LossBasedGrowth(
    patience=10,      # Wait 10 epochs before growing
    cooldown=5,       # Wait 5 epochs after growth
    min_delta=1e-4    # Minimum improvement threshold
)

# Create trainer
trainer = DENTrainer(
    network=network,
    growth_strategy=growth_strategy,
    optimizer=torch.optim.Adam,  # Pass optimizer class directly
    learning_rate=0.001
)

# Train with automatic growth
history = trainer.train(
    train_loader=train_loader,
    epochs=100,
    enable_growth=True
)

print(f"Final architecture: {network.get_layer_sizes()}")
print(f"Total parameters: {network.get_num_parameters()}")

# NEW: Plot with wall-clock time on x-axis
plot_training_history(history, save_path='training_time.png', use_time=True)
```

### Continual Learning

```python
# Initial training
trainer.train(train_loader_1, epochs=50)

# Learn from new data (network grows automatically if needed)
trainer.continual_learning(
    new_data=X_new,
    new_targets=y_new,
    epochs=20
)
```

### Save and Restart

```python
# Save checkpoint
trainer.save_checkpoint('checkpoint.pt')

# Later... restart training
trainer.load_checkpoint('checkpoint.pt')
trainer.train(new_train_loader, epochs=50)
```

## 📖 Core Components

### 1. DynamicExpandableNetwork

The main network class that supports dynamic architecture changes.

```python
network = DynamicExpandableNetwork(
    input_size=10,           # Number of input features
    output_size=1,           # Number of outputs
    hidden_sizes=[32, 32],   # Initial hidden layer sizes
    activation=nn.ReLU,      # Activation function (nn.Module class)
    dropout=0.1,            # Dropout rate (optional)
    task_type='regression'  # 'regression' or 'classification'
)

# You can also use:
# activation=nn.Tanh
# activation=nn.GELU
# activation=nn.LeakyReLU
# activation=lambda: nn.ReLU(inplace=True)
```

**Key Methods:**
- `expand_layer_width(layer_idx, num_neurons)`: Add neurons to a layer
- `expand_depth(position, num_neurons)`: Add a new layer
- `save_checkpoint(path)`: Save network state
- `load_checkpoint(path)`: Load network state

### 2. Growth Strategies

Control when and how the network grows.

#### LossBasedGrowth

Grows when loss plateaus:

```python
from den import LossBasedGrowth

strategy = LossBasedGrowth(
    patience=15,                    # Epochs to wait for improvement
    cooldown=5,                     # Epochs to wait after growth
    min_delta=1e-4,                # Improvement threshold
    width_growth_ratio=0.5,        # Add 50% more neurons
    max_neurons_per_expansion=32   # Max neurons to add at once
)
```

#### GradientBasedGrowth

Grows based on gradient magnitudes:

```python
from den import GradientBasedGrowth

strategy = GradientBasedGrowth(
    patience=10,
    gradient_threshold=0.1,  # Threshold for gradient magnitude
    max_neurons_per_expansion=32
)
```

#### AdaptiveGrowth

Combines multiple signals (recommended for general use):

```python
from den import AdaptiveGrowth

strategy = AdaptiveGrowth(
    patience=15,
    cooldown=10,
    loss_threshold=1e-4,
    gradient_threshold=0.05,
    max_network_size=10000  # Prevent unlimited growth
)
```

#### BiologicalGrowth

**NEW!** Mimics neural development in living organisms:

```python
from den import BiologicalGrowth

strategy = BiologicalGrowth(
    patience=12,
    cooldown=8,
    activity_threshold=0.3,      # Neuron activity trigger
    pruning_threshold=0.1,       # Prune weak neurons
    energy_cost_weight=0.01,     # Metabolic cost penalty
    max_neurons_per_expansion=16,
    enable_pruning=True,         # Enable synaptic pruning
    hebbian_window=5             # Activity correlation window
)
```

**Biological Principles:**
- **Activity-dependent neurogenesis**: Active neurons promote growth
- **Hebbian learning**: "Neurons that fire together wire together"
- **Synaptic pruning**: Removes weak/unused neurons
- **Energy efficiency**: Balances performance vs. network size
- **Competitive growth**: Resources allocated to active regions

Perfect for applications mimicking brain development and adaptive learning!

### 3. DENTrainer

Handles training with automatic growth.

```python
trainer = DENTrainer(
    network=network,
    growth_strategy=strategy,
    optimizer=torch.optim.AdamW,  # Pass optimizer class directly
    optimizer_kwargs={'weight_decay': 0.01},  # Optional optimizer parameters
    learning_rate=0.001,
    device='cuda',             # 'cuda' or 'cpu'
    verbose=True
)

# You can use any PyTorch optimizer:
# optimizer=torch.optim.Adam
# optimizer=torch.optim.SGD (with optimizer_kwargs={'momentum': 0.9})
# optimizer=torch.optim.RMSprop
```

**Key Methods:**
- `train(train_loader, epochs, enable_growth=True)`: Train with growth
- `evaluate(data_loader)`: Evaluate performance
- `continual_learning(new_data, new_targets)`: Learn from new data
- `predict(data)`: Make predictions

**New in v0.2:**
- Training history includes timestamps (`timestamps` field)
- Growth events include `timestamp` and `datetime` fields
- Plot training vs wall-clock time with `use_time=True`

## 📊 Visualization and Analysis

```python
from den.utils import (
    plot_training_history,
    print_growth_summary,
    visualize_network_architecture
)

# Plot training progress and growth events
plot_training_history(history, save_path='training.png')

# Print detailed growth summary
print_growth_summary(history)

# Visualize final architecture
visualize_network_architecture(network, save_path='architecture.png')
```

## 💡 Examples

### Example 1: Simple Regression

```bash
cd examples
python simple_regression.py
```

Demonstrates basic usage with synthetic regression data.

### Example 2: Continual Learning

```bash
python continual_learning.py
```

Shows how to learn multiple tasks sequentially with automatic growth.

### Example 3: Classification

```bash
python classification_example.py
```

Multi-class classification with dynamic architecture.

### Example 4: Biological Growth (NEW!)

```bash
python biological_growth_example.py
```

**Mimics how real brains develop!** Watch the network grow through developmental stages:
- **Infant stage**: Learning simple patterns with basic neural circuits
- **Child stage**: Developing complex representations
- **Adult stage**: Mastering abstract concepts

Features biological principles:
- Activity-dependent neurogenesis (neurons grow where needed)
- Hebbian learning (firing together = wiring together)
- Synaptic pruning (weak connections removed)
- Energy efficiency optimization

This is the most realistic simulation of how living brains learn and adapt!

## 🏗️ Architecture

```
Dynamically-Expandable-Network/
├── den/
│   ├── __init__.py           # Package initialization
│   ├── core.py               # DynamicExpandableNetwork class
│   ├── layers.py             # ExpandableLinear layer
│   ├── growth_strategy.py    # Growth decision strategies
│   ├── trainer.py            # DENTrainer class
│   └── utils.py              # Visualization utilities
├── examples/
│   ├── simple_regression.py
│   ├── continual_learning.py
│   └── classification_example.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## 📐 Growth Strategy Mathematics

This section provides detailed mathematical formulations for each growth strategy.

### LossBasedGrowth - Mathematical Formulation

**Core Principle**: Plateau Detection
Triggers growth when training loss stops improving for a sustained period.

**Decision Criteria:**

1. **Loss Improvement Check:**
   ```
   Δₜ = L_best - L_t

   where:
   - L_t: Current loss at epoch t
   - L_best: Best loss observed so far
   - δ: Minimum improvement threshold (min_delta)

   Improvement condition: Δₜ > δ
   ```

2. **Patience Counter:**
   ```
   p_t = {
       0,           if Δₜ > δ
       p_{t-1} + 1, otherwise
   }

   Growth trigger: p_t ≥ P
   where P = patience parameter
   ```

3. **Cooldown Period:**
   ```
   c_t = epochs since last growth

   Allow growth only if: c_t ≥ C
   where C = cooldown parameter
   ```

**Growth Magnitude:**

For **width expansion**:
```
n_new = min(⌊n_current × r⌋, n_max)

where:
- n_current: Current layer size
- r: width_growth_ratio (default 0.5)
- n_max: max_neurons_per_expansion
```

For **depth expansion** (triggered every `depth_threshold` width expansions):
```
n_new = min(mean(layer_sizes), n_max)
position = ⌊num_layers / 2⌋
```

**Layer Selection for Width Expansion:**
```
i* = argmin{|L_i|}

where L_i is the size of layer i
```
Choose smallest layer to balance architecture.

---

### GradientBasedGrowth - Mathematical Formulation

**Core Principle**: Under-Parameterization Detection
Monitors gradient magnitudes to identify layers struggling to learn.

**Gradient Tracking:**

1. **Layer-wise Gradient Norm:**
   ```
   g_i^(t) = ||∇_W_i L||₂

   where:
   - W_i: Weights of layer i
   - L: Loss function
   - || · ||₂: L2 norm
   ```

2. **Temporal Gradient History:**
   ```
   G_i = {g_i^(t-k), g_i^(t-k+1), ..., g_i^(t)}

   ḡ_i = (1/k) Σ g_i^(τ)

   where k = hebbian_window
   ```

3. **Network-Level Gradient:**
   ```
   ḡ = (1/N) Σᵢ ḡ_i

   where N = number of layers
   ```

**Growth Decision:**

Grow when BOTH conditions are met:

```
Condition 1: ḡ > θ_g  (high gradients)
Condition 2: p_t ≥ P  (loss plateau)

where:
- θ_g: gradient_threshold
- p_t: patience counter (as in LossBasedGrowth)
```

**Layer Selection:**
```
i* = argmax{ḡ_i}
```
Expand the layer with highest average gradient (most struggling).

**Growth Magnitude:**
```
n_new = min(max(⌊n_i* × 0.3⌋, 1), n_max)
```

---

### AdaptiveGrowth - Mathematical Formulation

**Core Principle**: Multi-Signal Integration
Combines loss, gradients, and network capacity for intelligent growth decisions.

**Loss Analysis:**

1. **Loss Standard Deviation:**
   ```
   σ_L = √[(1/P) Σ (L_t - L̄)²]

   where L̄ = (1/P) Σ L_t over last P epochs
   ```

2. **Loss Improvement Rate:**
   ```
   ΔL = L_{t-P} - L_t
   ```

3. **Stagnation Detection:**
   ```
   stagnant = (ΔL < θ_L) ∧ (σ_L < θ_L)

   where θ_L = loss_threshold
   ```

**Network Efficiency:**

```
E_t = 1 / (L_t × (1 + n_params/1000))

where:
- E_t: Efficiency at time t
- n_params: Total network parameters
```

Penalizes large networks with poor performance.

**Plasticity Detection:**

```
needs_plasticity = (ḡ > θ_g) ∨ (E_t < 0.8 × Ē)

where:
- ḡ: Average gradient norm
- θ_g: gradient_threshold
- Ē: Mean efficiency over recent epochs
```

**Growth Decision:**
```
grow = (stagnant ∨ needs_plasticity) ∧ (p_t ≥ P) ∧ (n_params < n_max)
```

**Intelligent Layer Selection:**

Combined score for each layer:
```
S_i = (a_i / (n_i + 1)) × (1 + α_i)

where:
- a_i: Activity/gradient score
- n_i: Current layer size
- α_i: Activation magnitude
```

**Growth Type Decision:**

```
type = {
    depth,  if (N < 3) ∨ (Var(layer_sizes) < μ × 0.1)
    width,  otherwise
}

where:
- N: Number of layers
- μ: Mean layer size
- Var: Variance of layer sizes
```

**Depth Growth:**
```
position = ⌊N/2⌋
n_new = min(μ, n_max)
```

**Width Growth:**
```
i* = argmax{S_i}
n_new = min(⌊n_i* × 0.5⌋, n_max)
```

---

### BiologicalGrowth - Mathematical Formulation

**Core Principle**: Biomimetic Neural Development
Simulates biological neurogenesis, pruning, and metabolic constraints.

**Neuron Activity (Firing Rate Analog):**

1. **Layer Activity:**
   ```
   A_i = Σⱼ |w_ij| + |b_i|

   Normalized: â_i = A_i / max(A)

   where:
   - w_ij: Weight from neuron j to i
   - b_i: Bias of neuron i
   ```

2. **Temporal Activity:**
   ```
   Ā_i^(t) = (1/k) Σ_{τ=t-k}^t â_i^(τ)

   where k = hebbian_window
   ```

**Energy Cost (Metabolic Constraint):**

```
E_metabolic = (n_params^1.2) × w_energy

where:
- n_params: Total parameters
- w_energy: energy_cost_weight
- 1.2 exponent: Nonlinear scaling (like brain metabolism)
```

**Network Efficiency:**

```
η = 1 / (L × (1 + n_params/1000))

where:
- L: Current loss
- η: Efficiency (performance per parameter)
```

**Efficiency History:**

```
H_η = {η_{t-k}, ..., η_t}

Declining efficiency: η_t < 0.8 × mean(H_η)
```

**Plasticity Need Detection:**

```
plasticity_needed = (ḡ > 0.05) ∨ (declining_efficiency)

where ḡ = average gradient norm
```

**Activity-Dependent Growth:**

High sustained activity triggers neurogenesis:
```
high_activity = Ā_overall > θ_A

where:
- Ā_overall = mean(Ā_i) across all layers
- θ_A: activity_threshold
```

**Growth Decision (Hebbian-Inspired):**

```
grow = (plasticity_needed ∨ high_activity) ∧
       (p_t ≥ P) ∧
       (c_t ≥ C) ∧
       (n_params < n_max)
```

**Competitive Resource Allocation:**

Score for each layer (competitive growth):
```
S_i = (2 × Ā_i) + (1/(n_i + 1))

where:
- Ā_i: Activity score (favor active regions)
- n_i: Layer size (favor small layers)
```

Target layer: `i* = argmax{S_i}`

**Growth Magnitude (Activity-Dependent):**

```
n_base = max(⌊n_i* × 0.3⌋, 2)
n_bonus = ⌊n_base × Ā_i*⌋
n_new = min(n_base + n_bonus, n_max)
```

More active layers → more new neurons (like BDNF signaling).

**Depth Growth (Cortical Development):**

Triggered when activity is very high:
```
if max(Ā_i) > 0.7 and N < 6:
    Add layer at position ⌊N/2⌋
```

**Synaptic Pruning:**

1. **Neuron Importance:**
   ```
   I_j = ||w_·j||₂ + |b_j|

   Normalized: î_j = I_j / max(I)
   ```

2. **Pruning Criterion:**
   ```
   prune_j = (î_j < θ_prune) ∧ (n_i > n_min)

   where:
   - θ_prune: pruning_threshold
   - n_min: Minimum layer size (e.g., 8)
   ```

3. **Pruning Frequency:**
   ```
   Can prune if: (t - t_last_prune) > 2C

   where C = cooldown
   ```

**"Use it or Lose it" Principle:**

If >20% of layer's neurons have low activity:
```
weak_fraction = |{j : î_j < θ_prune}| / n_i

if weak_fraction > 0.2:
    Trigger pruning
```

---

## 🔬 How It Works

### Width Expansion

When a layer needs more capacity, DEN adds neurons:

1. Creates a new layer with additional neurons
2. Copies existing weights
3. Initializes new neuron weights (Kaiming/Xavier)
4. Updates subsequent layer input dimensions

### Depth Expansion

When the network needs more expressiveness, DEN adds layers:

1. Determines optimal position for new layer
2. Creates layer with appropriate dimensions
3. Adjusts connections to maintain gradient flow
4. Initializes weights to preserve learned knowledge

### Growth Triggers

Networks grow when:
- Loss stops improving (plateau detection)
- Gradients remain high (under-parameterization)
- Capacity utilization is high
- New patterns don't fit current architecture

## 🎯 Use Cases

### 1. Continual Learning
Learn from data streams without catastrophic forgetting:
```python
for task_data in data_stream:
    trainer.continual_learning(task_data, epochs=20)
```

### 2. Online Learning
Adapt to new patterns in real-time:
```python
while new_data_available():
    trainer.continual_learning(get_new_batch(), epochs=5)
```

### 3. AutoML
Automatically find optimal architecture:
```python
# Start small, let it grow to the right size
network = DynamicExpandableNetwork(
    input_size=features,
    output_size=targets,
    hidden_sizes=[8, 8]  # Very small initial size
)
```

### 4. Transfer Learning
Grow network for new tasks while preserving old knowledge.

## ⚙️ Advanced Configuration

### Custom Growth Strategy

```python
from den.growth_strategy import GrowthStrategy

class MyCustomGrowth(GrowthStrategy):
    def should_grow(self, metrics, network, epoch):
        # Your custom logic
        if metrics['loss'] > threshold:
            return True, "Custom reason"
        return False, None

    def determine_growth_action(self, metrics, network):
        # Decide how to grow
        return {
            'type': 'width',
            'layer_idx': 0,
            'num_neurons': 16
        }
```

### Custom Loss Function

```python
import torch.nn as nn

trainer = DENTrainer(
    network=network,
    loss_function=nn.HuberLoss(),  # Custom loss
    ...
)
```

## 📈 Performance Tips

1. **Start Small**: Begin with small architectures (8-16 neurons)
2. **Tune Patience**: Higher patience = fewer, larger growths
3. **Set Max Size**: Use `max_network_size` to prevent unlimited growth
4. **Use Adaptive Strategy**: Best for most use cases
5. **Monitor Growth**: Use visualization tools to understand growth patterns

## 🔍 Monitoring

```python
# During training
print(f"Current architecture: {network.get_layer_sizes()}")
print(f"Total parameters: {network.get_num_parameters()}")
print(f"Growth events: {len(history['growth_events'])}")

# Layer importance analysis
from den.utils import analyze_layer_importance
importance = analyze_layer_importance(network)
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=den --cov-report=html
```

## 📚 Research Background

This implementation is inspired by:
- Dynamic Expandable Networks for continual learning
- Progressive Neural Networks
- Neural Architecture Search
- Lifelong learning systems

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional growth strategies
- Pruning capabilities
- More visualization tools
- Additional examples
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Research community for continual learning insights

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Submit a pull request

## 🗺️ Roadmap

- [ ] Pruning capabilities
- [ ] Multi-task learning support
- [ ] Automated hyperparameter tuning
- [ ] More growth strategies
- [ ] Integration with popular datasets
- [ ] Distributed training support
- [ ] ONNX export support

---

**Start with a small network and let it grow!** 🌱➡️🌳

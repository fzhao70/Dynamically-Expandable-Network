# Dynamically Expandable Network (DEN)

A PyTorch implementation of **Dynamically Expandable Neural Networks** that can automatically grow their architecture during training to accommodate complex patterns and new data.

## 🌟 Features

- **Automatic Growth**: Networks expand automatically based on training dynamics
- **Width Expansion**: Add neurons to existing layers when needed
- **Depth Expansion**: Add new layers to increase network capacity
- **Continual Learning**: Learn from new data streams without forgetting
- **Multiple Growth Strategies**: Choose from loss-based, gradient-based, or adaptive growth
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
from torch.utils.data import DataLoader, TensorDataset
from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth

# Create your data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 1)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# Create a DEN with initial small architecture
network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[16, 16],  # Start small!
    activation='relu',
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
    optimizer='adam',
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
    activation='relu',       # Activation function
    dropout=0.1,            # Dropout rate (optional)
    task_type='regression'  # 'regression' or 'classification'
)
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

Combines multiple signals (recommended):

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

### 3. DENTrainer

Handles training with automatic growth.

```python
trainer = DENTrainer(
    network=network,
    growth_strategy=strategy,
    optimizer='adam',           # 'adam', 'sgd', or 'rmsprop'
    learning_rate=0.001,
    device='cuda',             # 'cuda' or 'cpu'
    verbose=True
)
```

**Key Methods:**
- `train(train_loader, epochs, enable_growth=True)`: Train with growth
- `evaluate(data_loader)`: Evaluate performance
- `continual_learning(new_data, new_targets)`: Learn from new data
- `predict(data)`: Make predictions

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

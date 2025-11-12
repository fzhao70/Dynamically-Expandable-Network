# API Migration Guide - DEN v0.2.0

## Overview

This guide helps you migrate from the old string-based API to the new direct class/module API.

## Key Changes

### 1. Activation Functions - Use `nn.Module` Instead of Strings

**Before (v0.1.0):**
```python
network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[32, 32],
    activation='relu',  # String
    task_type='regression'
)
```

**After (v0.2.0):**
```python
network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[32, 32],
    activation=nn.ReLU,  # Pass class directly
    task_type='regression'
)
```

**Alternative Options:**
```python
# Option 1: Pass the class
activation=nn.ReLU

# Option 2: Pass an instance
activation=nn.ReLU()

# Option 3: Pass a lambda
activation=lambda: nn.LeakyReLU(negative_slope=0.2)

# Option 4: Use any custom activation
activation=nn.GELU
```

### 2. Optimizers - Use `optim.Class` Instead of Strings

**Before (v0.1.0):**
```python
trainer = DENTrainer(
    network=network,
    optimizer='adam',  # String
    learning_rate=0.001
)
```

**After (v0.2.0):**
```python
trainer = DENTrainer(
    network=network,
    optimizer=torch.optim.Adam,  # Pass class
    learning_rate=0.001
)
```

**With Additional Parameters:**
```python
trainer = DENTrainer(
    network=network,
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={'weight_decay': 0.01, 'betas': (0.9, 0.999)},
    learning_rate=0.001
)
```

### 3. New Feature: Timestamp-Based Plotting

**Epoch-based plot (default):**
```python
plot_training_history(history, save_path='training.png')
```

**Time-based plot (NEW!):**
```python
plot_training_history(history, save_path='training_time.png', use_time=True)
```

This shows wall-clock time on the x-axis instead of epochs, which is useful for:
- Comparing training efficiency
- Understanding real-world training duration
- Analyzing growth timing in production systems

### 4. Growth Events Now Include Timestamps

Growth events in the training history now include:
- `timestamp`: Seconds from training start
- `datetime`: Human-readable ISO timestamp

```python
for event in history['growth_events']:
    print(f"Epoch {event['epoch']}: {event['timestamp']:.2f}s - {event['datetime']}")
```

## Complete Migration Example

**Before:**
```python
import torch
from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth

network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[16, 16],
    activation='relu',
    task_type='regression'
)

trainer = DENTrainer(
    network=network,
    growth_strategy=LossBasedGrowth(),
    optimizer='adam',
    learning_rate=0.001
)

history = trainer.train(train_loader, epochs=100)
```

**After:**
```python
import torch
import torch.nn as nn
from den import DynamicExpandableNetwork, DENTrainer, LossBasedGrowth
from den.utils import plot_training_history

network = DynamicExpandableNetwork(
    input_size=10,
    output_size=1,
    hidden_sizes=[16, 16],
    activation=nn.ReLU,  # Changed
    task_type='regression'
)

trainer = DENTrainer(
    network=network,
    growth_strategy=LossBasedGrowth(),
    optimizer=torch.optim.Adam,  # Changed
    learning_rate=0.001
)

history = trainer.train(train_loader, epochs=100)

# NEW: Timestamp-based plotting
plot_training_history(history, save_path='epochs.png')
plot_training_history(history, save_path='time.png', use_time=True)
```

## Benefits

### 1. **More Flexible**
You can now use any PyTorch activation or optimizer, including custom ones:
```python
# Custom activation
class MyActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

network = DynamicExpandableNetwork(..., activation=MyActivation)
```

### 2. **More Pythonic**
Passing classes directly is more idiomatic Python than string magic.

### 3. **Better Type Checking**
IDEs and type checkers can now validate your activation/optimizer choices.

### 4. **Easier Configuration**
Optimizer parameters are now explicit:
```python
trainer = DENTrainer(
    optimizer=torch.optim.SGD,
    optimizer_kwargs={'momentum': 0.9, 'nesterov': True},
    learning_rate=0.01
)
```

### 5. **Time-Aware Training**
New timestamp tracking lets you analyze training in wall-clock time, not just epochs.

## Backward Compatibility

Checkpoints from v0.1.0 can still be loaded in v0.2.0. The `load_checkpoint` method handles both old and new formats automatically.

## Questions?

See the updated examples in `/examples/` for complete working code.

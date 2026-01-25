# Neural-Symbolic Examples

Neural-symbolic training examples demonstrating integration where neural network outputs become probabilistic facts in logic programs.

## Overview

These examples showcase XLOG's v0.4.0-alpha neural-symbolic capabilities:

- **Neural Predicates**: `nn(network, [inputs], output, [labels]) :: predicate(args).`
- **Network Registration**: PyTorch networks with optimizers and schedulers
- **Tensor Sources**: External data (images, embeddings) indexed by predicates
- **Differentiable Training**: Gradients flow from logic queries through d-DNNF circuits to neural networks
- **Circuit Caching**: Compiled circuits reused across training iterations (100x+ speedup)

## Examples

### 01_minimal — MNIST Addition

Train a CNN to classify MNIST digits using **only addition supervision**.

```
examples/neural/01_minimal/
├── train.py       # Training script
└── data/          # MNIST data (auto-downloaded)
```

**The key insight**: The network never sees digit labels during training. Instead, it learns from queries like "What is the probability that image[i] + image[j] = 7?"

**Run:**
```bash
cd examples/neural/01_minimal
python train.py --epochs 50 --batch-size 32
```

**Program:**
```prolog
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
```

**Training:**
```python
# Query: What's P(image[0] + image[1] = 7)?
queries = ["addition(0, 1, 7)", "addition(2, 3, 5)", ...]

# NLL loss trains the network to maximize P(correct_sum)
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
```

## Planned Examples

The following neural-symbolic examples are planned for v0.4.0-beta:

| Example | Description |
|---------|-------------|
| **02_coins** | Two coin classifiers with win/loss logic |
| **03_mnist_multidigit** | Multi-digit number recognition |
| **04_hwf** | Handwritten Formula recognition |
| **05_poker** | Card rank classification |
| **06_clutrr** | Family relationship reasoning with knowledge graphs |

## How Neural-Symbolic Training Works

1. **Define neural predicates** — declare that a network provides probability distributions
2. **Forward pass** — network outputs become weights in annotated disjunctions
3. **Knowledge compilation** — logic program compiled to d-DNNF circuit (cached!)
4. **Weighted model counting** — circuit evaluated for query probability
5. **Backward pass** — gradients flow from NLL loss through circuit to networks
6. **Optimizer step** — PyTorch updates network weights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision (for MNIST)
- pyxlog (built with `maturin develop --release`)

## API Reference

```python
import pyxlog

# Compile program with neural predicates
program = pyxlog.Program.compile("""
    nn(net_name, [X], Y, [0,1,...,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

# Register PyTorch network
program.register_network("net_name", model, optimizer, scheduler=None)

# Add tensor data source
program.add_tensor_source("train", images_tensor)

# Single query training
loss = program.forward_backward("addition(0, 1, 7)")  # host scalar (convenience)

# Strict GPU-native training (returns CUDA tensor loss; no host reads required)
loss_t = program.forward_backward_tensor("addition(0, 1, 7)")

# Batch training
stats = program.train_epoch(queries, batch_size=32)
avg_loss = stats.avg_loss

# Full training loop
history = pyxlog.train_model(
    program,
    queries,
    epochs=50,
    batch_size=32,
    log_iter=10
)

# Access training history
print(f"Final loss: {history.epoch_losses[-1]}")
```

## References

- [XLOG Design Doc](../../docs/plans/2026-01-20-v0.4.0-neural-symbolic-design.md) — v0.4.0 neural-symbolic design
- [Implementation Plan](../../docs/plans/v0.4.0-alpha-implementation.md) — v0.4.0-alpha implementation details

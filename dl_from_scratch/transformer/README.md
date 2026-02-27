# Transformer

Transformer implemented from scratch in two ways:

| Variant | Description |
|---------|-------------|
| [numpy_based](./numpy_based/README.md) | Pure NumPy with manually derived gradients |
| [pytorch_based](./pytorch_based/README.md) | PyTorch primitives with autograd |

The NumPy variant implements everything from scratch — forward pass, backpropagation, and all gradient computations. Mathematical derivations for each component are linked from its [README](./numpy_based/README.md).
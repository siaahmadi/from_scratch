# Shared utilities

This submodule contains the functions commonly used by the models for forward and backward passes. It also contains helpers to load the training data.

| Module | Function | Description | Derivations (if any) |
|---------|-------------|-------------|-------------|
| `helpers` | `softmax` | The numerically stable softmax function | [PDF](./softmax_gradient.pdf) | |
| `helpers` | `masked_softmax` | Allows applying a mask to `softmax`; always returns finite values even if entire row masked | |
| `helpers` | `cross_entropy_loss` | The cross entropy loss function | |
| `helpers` | `cross_entropy_gradient` | The gradient of the cross entropy loss function | |
| `helpers` | `loss_and_accuracy` | Compute both cross entropy loss and the accuracy of a classifier | |
| `helpers` | `xavier_uniform` | Initialie a matrix using the Xavier uniform scheme | [Paper](https://proceedings.mlr.press/v9/glorot10a.html) |
| `helpers` | `xavier_normal` | Initialie a matrix using the Xavier normal scheme | [Paper](https://proceedings.mlr.press/v9/glorot10a.html) |


# Deep Learning from Scratch

Implementations of deep learning primitives using only NumPy or by re-building PyTorch modules from lower-level PyTorch operations.

| Module | Description |
|--------|-------------|
| [mlp](./dl_from_scratch/mlp/README.md) | Multi-layer perceptron |
| [transformer](./dl_from_scratch/transformer/README.md) | Multi-head attention and transformer blocks; **contains a clear explanation on the transformer attention mechanism** |
| [utils](./dl_from_scratch/utils/README.md) | Shared utilities |

# A heads-up on notation

Throughout this code base, the matrix operations assume the number of rows represents the dimensionality of the input and the number columns that of the output. In the case of the input layer, the "dimensionality of the input" becomes the "number of inputs."

In addition, the gradients are computed in the denominator layout so that the gradient of the loss with respect to each parameter has the same shape as the parameter itself.

This is the format followed both in the math (derivations in the PDF files) as well as the code itself.

In the PDFs, lowercase boldface letters represent vectors and uppercase boldface letters represent matrices, as is typical in math textbooks. Italic lowercases represent scalars (single numbers).
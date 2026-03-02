# Deep Learning from Scratch

Implementations of deep learning models using only basic NumPy operations only.

This is primarily intended as an excercise, but is also written for those who are generally familiar with the models but might need some re-orientation or reminders. Make sure to review the [mathematical derivations](derivations_list.md) which go together with the code.

Analogs of a few PyTorch modules are also rebuilt here from lower-level PyTorch operations.

| Module | Description |
|--------|-------------|
| [mlp](./dl_from_scratch/mlp/) | Multi-layer perceptron |
| [transformer](./dl_from_scratch/transformer/) | Multi-head attention and transformer blocks; **contains a clear explanation on the transformer attention mechanism** |
| [utils](./dl_from_scratch/utils/) | Shared utilities |
| [list of derivations](derivations_list.md) | List of step by step derivation of each component

# A heads-up on notation

(This doesn't render well on Github unfortunately, but hopefully you get the point).

Throughout this code base, the matrix operations assume the number of rows represents the dimensionality of the input and the number columns that of the output. In the case of the input layer, the "dimensionality of the input" becomes the "number of inputs."

In addition, the gradients are computed in the denominator layout so that the gradient of the loss with respect to each parameter has the same shape as the parameter itself.

This is the format followed both in the math (derivations in the PDF files) as well as the code itself.

In the PDFs, lowercase boldface letters (like $\mathbf{x}$) represent vectors and uppercase boldface letters represent matrices (like $\mathbf{X}$), as is typical in math textbooks. Italic lowercases represent scalars (single numbers, like $x$).
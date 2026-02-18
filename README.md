# Two-Layer MLP from Scratch

A minimal implementation of a two-layer neural network (multi-layer perceptron) built from scratch using only NumPy (no deep learning frameworks)

The network uses ReLU activation with softmax output and cross-entropy loss. The weights are initialized using the Kaiming initialization. The backpropagation and gradient computation are implemented manually. A vanilla SGD optimizer (with learning rate decay) is build into the model class.

See `from_scratch_2layer_mlp.pdf` for complete mathematical derivations.

## Structure

```
├── two_layer_nn.py          # Neural network class with forward/backward pass
├── helpers.py               # Softmax, cross-entropy, accuracy functions
├── mnist_data.py            # MNIST data loading and preprocessing
├── train.py                 # Training loop
└── from_scratch_2layer_mlp.pdf  # Mathematical derivations
```

## Architecture

**Input:** 784 (28×28 MNIST images)  
**Hidden Layer:** 128 units with ReLU activation  
**Output:** 10 classes with softmax

## Usage

```bash
python train.py
```

**Hyperparameters** (in `train.py`):
```python
{
    'input_size': 784,
    'hidden_size': 128,
    'output_size': 10,
    'learning_rate': 0.1,
    'lr_decay': 0.99,
    'n_epochs': 10,
    'batch_size': 32,
}
```

## Results

- **Training accuracy:** ~96%
- **Validation accuracy:** ~97%
- **Test accuracy:** ~97%


## Dependencies

```bash
numpy
scikit-learn  # for MNIST data loading only
```

## License

MIT

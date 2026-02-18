# Two-Layer MLP from Scratch

A minimal implementation of a two-layer neural network (multi-layer perceptron) built from scratch using only NumPy (no deep learning frameworks).

The network uses ReLU activation with softmax output and cross-entropy loss. The weights are initialized using the Kaiming initialization. The backpropagation and gradient computation are implemented manually. A vanilla SGD optimizer (with learning rate decay) is build into the model class.

See `derivations.pdf` for complete mathematical derivations.

Set hyperparameters and run training in `train.py`.

## Structure

```
├── two_layer_nn.py  # Neural network class with forward/backward pass
├── helpers.py       # Softmax, cross-entropy, accuracy functions
├── mnist_data.py    # MNIST data loading and preprocessing
├── train.py         # Training loop
└── derivations.pdf  # Mathematical derivations
```

## Architecture

**Input:** 784 (28×28 MNIST images)  
**Hidden Layer:** 256 units with ReLU activation  
**Output:** 10 classes with softmax

## Usage

```bash
python train.py
```

## Results

- **Training accuracy:** ~98.9% (after 10 epochs), ~~99.7% (after 20 epochs)
- **Validation accuracy:** ~97.6% (after 10 epochs), ~~98.0% (after 20 epochs)
- **Test accuracy:** ~97.5% (after 10 epochs), ~97.9% (after 20 epochs)


## Dependencies

```bash
numpy
scikit-learn  # for MNIST data loading only
```

## License

MIT

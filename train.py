import numpy as np

from two_layer_nn import TwoLayerMLP
from mnist_data import X_train, y_train, X_valid, y_valid, X_test, y_test
from helpers import cross_entropy_loss, cross_entropy_gradient, accuracy


hyperparam = {
    'dtype': np.float64,
    'input_size': X_train.shape[1],
    'hidden_size': 128,
    'output_size': y_train.shape[1],
    'learning_rate': 10e-2,
    'lr_decay': 0.99,
    'n_epochs': 10,
    'batch_size': 32,
}

model = TwoLayerMLP(
    hyperparam['input_size'],
    hyperparam['hidden_size'],
    hyperparam['output_size'],
    hyperparam['learning_rate'],
    dtype=hyperparam['dtype'],
)

X_train = X_train.astype(hyperparam['dtype'])
y_train = y_train.astype(hyperparam['dtype'])
X_valid = X_valid.astype(hyperparam['dtype'])
y_valid = y_valid.astype(hyperparam['dtype'])
X_test  = X_test.astype( hyperparam['dtype'])
y_test  = y_test.astype( hyperparam['dtype'])


n_batches = np.ceil(X_train.shape[0] // hyperparam['batch_size']).astype(int)

for epoch in range(1, 1 + hyperparam['n_epochs']):
    for batch in range(n_batches):
        X_batch = X_train[batch * hyperparam['batch_size'] : (1 + batch) * hyperparam['batch_size']]
        y_batch = y_train[batch * hyperparam['batch_size'] : (1 + batch) * hyperparam['batch_size']]

        logits = model.forward(X_batch)
        gradient = cross_entropy_gradient(logits, y_batch)
        model.backward(gradient)
        model.step()

        if batch % 100 == 0:
            logits = model.forward(X_train)
            loss = cross_entropy_loss(logits, y_train)
            accuracy_train = accuracy(logits, y_train)

            logits_valid = model.forward(X_valid)
            loss_valid = cross_entropy_loss(logits_valid, y_valid)
            accuracy_valid = accuracy(logits_valid, y_valid)

            print(f"Epoch {epoch}; batch {1 + batch}, train loss = {loss.mean():.3f} ({accuracy_train*100:.1f}%), valid loss = {loss_valid.mean():.3f} ({accuracy_valid*100:.1f}%)")

    model.lr *= hyperparam['lr_decay']


logits_valid = model.forward(X_valid)
loss_valid = cross_entropy_loss(logits_valid, y_valid)
accuracy_valid = accuracy(logits_valid, y_valid)

print("Training done.")
print(f"Accuracy on validation set: {accuracy_valid*100:.1}%")
print(f"Loss on validation set: {loss_valid:.4f}")
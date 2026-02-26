import numpy as np
import copy

from two_layer_nn import TwoLayerMLP
from dl_from_scratch.utils.mnist_data import X_train, y_train, X_valid, y_valid, X_test, y_test
from dl_from_scratch.utils.helpers import softmax, cross_entropy_gradient, loss_and_accuracy


hyperparam = {
    'dtype': np.float64,
    'input_size': X_train.shape[1],
    'hidden_size': 256,
    'output_size': y_train.shape[1],
    'learning_rate': 1e-1,
    'lr_decay': 0.99,
    'n_epochs': 10,
    'batch_size': 32,
    'random_seed': 0,
}

model = TwoLayerMLP(
    hyperparam['input_size'],
    hyperparam['hidden_size'],
    hyperparam['output_size'],
    hyperparam['learning_rate'],
    dtype=hyperparam['dtype'],
    seed=hyperparam['random_seed'],
)

X_train = X_train.astype(hyperparam['dtype'])
y_train = y_train.astype(hyperparam['dtype'])
X_valid = X_valid.astype(hyperparam['dtype'])
y_valid = y_valid.astype(hyperparam['dtype'])
X_test  = X_test.astype( hyperparam['dtype'])
y_test  = y_test.astype( hyperparam['dtype'])


n_batches = np.ceil(X_train.shape[0] // hyperparam['batch_size']).astype(int)
best_loss = np.inf

for epoch in range(1, 1 + hyperparam['n_epochs']):
    for batch in range(n_batches):
        X_batch = X_train[batch * hyperparam['batch_size'] : (1 + batch) * hyperparam['batch_size']]
        y_batch = y_train[batch * hyperparam['batch_size'] : (1 + batch) * hyperparam['batch_size']]

        logits = model(X_batch)
        q = softmax(logits)
        gradient = cross_entropy_gradient(q, y_batch)
        model.backward(gradient)
        model.step()

        if batch % 100 == 0:
            loss_train, accuracy_train = loss_and_accuracy(model, X_train, y_train)
            loss_valid, accuracy_valid = loss_and_accuracy(model, X_valid, y_valid)

            print(f"Epoch {epoch:>2}; batch {1 + batch:>4}, train loss = {loss_train:.3f} ({accuracy_train*100:.1f}%), valid loss = {loss_valid:.3f} ({accuracy_valid*100:.1f}%)")

    loss_valid, accuracy_valid = loss_and_accuracy(model, X_valid, y_valid)
    if loss_valid < best_loss:
        best_model = copy.deepcopy(model)
        best_loss = loss_valid
    
    model.lr *= hyperparam['lr_decay']

loss_train, accuracy_train = loss_and_accuracy(best_model, X_train, y_train)
loss_valid, accuracy_valid = loss_and_accuracy(best_model, X_valid, y_valid)
loss_test , accuracy_test  = loss_and_accuracy(best_model, X_test , y_test)

print()
print("Training done.")

print()
print(f"Accuracy on training set: {accuracy_train*100:.1f}%")
print(f"Loss on training set: {loss_train:.4f}")

print()
print(f"Accuracy on validation set: {accuracy_valid*100:.1f}%")
print(f"Loss on validation set: {loss_valid:.4f}")

print()
print(f"Accuracy on test set: {accuracy_test*100:.1f}%")
print(f"Loss on test set: {loss_test:.4f}")
print()
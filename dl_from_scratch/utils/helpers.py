import numpy as np

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy_loss(logits, p):
    logits -= np.max(logits, axis=1, keepdims=True)

    correct_class = np.argmax(p, axis=1)
    row = np.arange(logits.shape[0])
    return np.log(np.exp(logits).sum(axis=1)) - logits[row, correct_class]

def cross_entropy_gradient(q, p):
    return q - p

def loss_and_accuracy(model, X, true_label):

    logits = model(X)

    loss = cross_entropy_loss(logits, true_label).mean()

    predicted = np.argmax(softmax(logits), axis=1)
    true = np.argmax(true_label, axis=1)
    accuracy = np.mean(predicted == true)
    
    return loss, accuracy

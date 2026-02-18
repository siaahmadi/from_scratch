import numpy as np

def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(keepdims=True)

def cross_entropy_loss(q, p):
    correct_class = np.argmax(p, axis=1)
    row = np.arange(q.shape[0])
    return np.log(np.exp(q).sum(axis=1)) - q[row, correct_class]

def cross_entropy_gradient(q, p):
    return q - p

def accuracy(logits, true_class):
    probs = np.argmax(softmax(logits), axis=1)
    true = np.argmax(true_class, axis=1)
    
    return np.mean(probs == true)

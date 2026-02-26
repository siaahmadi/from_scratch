import numpy as np

def softmax(x, axis=None):
    x -= np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp_x[sum_exp_x==0] = 1 # avoid div-by-zero
    return exp_x / sum_exp_x

def masked_softmax(x, mask, axis=None):
    if mask is not None:
        mask = mask.astype(bool)
        x[mask] = -np.inf
    return softmax(x, axis=axis)

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

def xavier_uniform(shape, seed=None):
    rng = np.random.default_rng(seed=seed)
    d_in = shape[-2]
    var_factor = 1/np.sqrt(d_in)
    return rng.uniform(-1, 1, size=shape) * var_factor

def xavier_normal(shape, seed=None):
    rng = np.random.default_rng(seed=seed)
    d_in = shape[-2]
    var_factor = 1/np.sqrt(d_in)
    return rng.standard_normal(shape) * var_factor


if __name__ == "__main__":
    seed = 0

    rng = np.random.default_rng(seed=seed)

    N, D = 8, 4

    x = rng.standard_normal((N, D))

    mask = np.ones(x.shape, dtype=np.bool)

    y = masked_softmax(x, mask=mask, axis=1)

    print(y)
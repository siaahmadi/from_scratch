import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.weight = np.ones((d,))
        self.bias = np.zeros((d,))

        self.eps = eps
        self._grad = {
            'out_bias': np.ones_like(self.bias),
        }

    def forward(self, x):
        mu  = np.mean(x, axis=-1, keepdims=True)
        var = np.var (x, axis=-1, keepdims=True)

        xhat = (x - mu) / np.sqrt(var + self.eps)
        y = xhat * self.weight + self.bias

        self._grad['out_weight'] = xhat
        self._grad['out_xhat']   = self.weight

        return y
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self._grad['out_bias'])
        self._grad['grad_weight'] = np.sum(grad * self._grad['out_weight'], axis=0, keepdims=True)
        self._grad['grad_bias']   = np.sum(grad * self._grad['out_bias'], axis=0, keepdims=True)
        self._grad['grad_xhat']   = np.sum(grad * self._grad['out_xhat'], axis=0, keepdims=True)

        return self._grad['grad_x'] # todo: this gets involved...
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


if __name__ == "__main__":
    random_seed = 0
    rng = np.random.default_rng(seed=random_seed)

    N = 8
    D = 4

    empirical_mean, empirical_std = rng.uniform(-10, 10, size=(N, 1)), rng.uniform(-10, 10, size=(N, 1))

    x = rng.standard_normal((N, D)) * empirical_std + empirical_mean
    empirical_mean, empirical_std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    y = np.empty((N, D))
    y[:] = x
    
    LN = LayerNorm(D)
    LN_torch = nn.LayerNorm(D, dtype=torch.float64)

    x_ln = LN(x)
    x_ln_torch = LN_torch(torch.tensor(x))

    mean_0 = np.vstack((y.mean(axis=0), x_ln.mean(axis=0)))
    std_0  = np.vstack((y.std(axis=0), x_ln.std(axis=0)))
    mean_1 = np.hstack((y.mean(axis=1,keepdims=True), x_ln.mean(axis=1,keepdims=True)))
    std_1  = np.hstack((y.std(axis=1,keepdims=True), x_ln.std(axis=1,keepdims=True)))
    
    print('\n'*4)
    print(empirical_mean)
    print(empirical_std)
    print()
    print(mean_0)
    print(std_0)
    print()
    print(mean_1)
    print(std_1)
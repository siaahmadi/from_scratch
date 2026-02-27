import numpy as np


class EmbeddingLayer():
    def __init__(self, dict_size, d_embed):
        self.weight = np.random.standard_normal((dict_size, d_embed))
    
    def forward(self, idx):
        assert idx.dtype == np.dtype(int)
        return self.weight[idx]

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

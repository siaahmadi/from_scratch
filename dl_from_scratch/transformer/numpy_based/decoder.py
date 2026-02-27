import numpy as np

from dl_from_scratch.utils.helpers import softmax


def initialize_matrix(m, n):
    pass

class EmbeddingLayer():
    def __init__(self, n, d):
        self.weight = np.random.standard_normal((n, d))
    
    def forward(self, idx):
        assert idx.dtype == np.dtype(int)
        return self.weight[idx]

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class SelfAttention():
    def __init__(self, d, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v

        self.Q = initialize_matrix(d, d_k)
        self.K = initialize_matrix(d, d_k)
        self.V = initialize_matrix(d, d_v)

        self.variance = 1/d_k

    def forward(self, x):
        return SelfAttention._forward_with_matrices(x, self.K, self.Q, self.V, self.variance)
    
    def _forward_with_matrices(x, K, Q, V, variance):

        Q = x @ Q
        K = x @ K

        normalization_factor = np.sqrt(variance)

        scores = softmax(Q @ K.T)

        scaled_dot_product = normalization_factor * (scores @ V)

        return scaled_dot_product

class MultiHeadSelfAttention():
    def __init__(self, d_k=512, d_v=64, h=8):
        self.d_k = d_k
        self.d_v = d_v

        self.SelfAttentionCopies = [SelfAttention(d_k, d_v) for _ in range(h)]
    
    def forward(self, x):
        K = np.vstack([selfattention.K for selfattention in self.SelfAttentionCopies])
        Q = np.vstack([selfattention.Q for selfattention in self.SelfAttentionCopies])
        V = np.vstack([selfattention.V for selfattention in self.SelfAttentionCopies])
        variance = 1/self.d_k

        return SelfAttention._forward_with_matrices(x, K, Q, V, variance)

class MLP():
    pass

class TransformerLayer:
    def __init__(self, d_k=512, d_v=64, h=8):
        self.d_k = d_k
        self.d_v = d_v
        self.h   = h
        
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(h)
        self.MLP = MLP(d_k, d_k)
    
    def forward(self, x):
        attention = self.MultiHeadSelfAttention(x)
        out       = self.MLP(attention)

        return out

class LayerNorm():
    pass

class Transformer():
    def __init__(self, d_k=512, d_v=64, h=8):
        self.d_k = d_k
        self.d_v = d_v
        self.h   = h

        self.Embedding = EmbeddingLayer()
        self.MHSA = MultiHeadSelfAttention(h)
        self.LayerNorm = LayerNorm()
import numpy as np
from dl_from_scratch.utils.helpers import xavier_uniform, masked_softmax


def scaled_dot_product(q, k, upstream_grad=None):

    d_k = q.shape[-1]
    s = np.sqrt(1./d_k)

    k_T = np.moveaxis(k, -1, -2)

    S = (q @ k_T) * s # (N, L_q, L_k)

    grad = {
        'q': upstream_grad @ k * s if upstream_grad is not None else k * s,
        'k': np.moveaxis(upstream_grad, -1, -2) @ q * s if upstream_grad is not None else q * s,
    }

    return S, grad

def masked_attention(q, k, v, mask, upstream_grad=None):

    S = scaled_dot_product(q, k)
    A = masked_softmax(S, mask, axis=-1)
    a = A @ v # (N, L_k, d_k)

    da_dA = np.moveaxis(v, -1, -2)
    da_dv = np.moveaxis(A, -1, -2)

    # this design has two problems:
    # 1) inconsistent output formats
    # 2) won't cache forward results
    # TODO: best to convert to returning a function that can compute backward pass
    _, da_dS  = masked_softmax(S, mask, axis=-1, compute_grad=True, upstream_grad=da_dA)
    _, da_dqk = scaled_dot_product(q, k, compute_grad=True, upstream_grad=da_dS)

    grad = {
        'q': upstream_grad @ da_dqk['q'] if upstream_grad is not None else da_dqk['q'],
        'k': upstream_grad @ da_dqk['k'] if upstream_grad is not None else da_dqk['k'],
        'v': upstream_grad @ da_dv if upstream_grad is not None else da_dv,
    }

    return a, grad

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, kdim=None, vdim=None, seed=None):
        """Expects a batch of shape (N, L, d_model) where
            N: batch size
            L: sequence length
        
        Also handles unbatched input shape (L, d_model)
        """
        
        assert d_model % num_heads == 0, "Model (embedding) dimension must be divisible by the number of heads."

        self.d_model   = d_model
        self.num_heads = num_heads
        d_k = d_model // num_heads

        if kdim is None:
            kdim = d_model
        
        if vdim is None:
            vdim = d_model
        
        self.Wq = xavier_uniform((num_heads, d_model , d_k), seed=seed)
        self.Wk = xavier_uniform((num_heads, kdim    , d_k), seed=seed)
        self.Wv = xavier_uniform((num_heads, vdim    , d_k), seed=seed)
        self.Wo = xavier_uniform((d_model  , d_model      ), seed=seed)
    
    def forward(self, q, k, v, mask=None):
        """q, k, and v all of shape (N, L, d_model)"""

        grad = {}

        Q = np.dot(q, self.Wq)
        K = np.dot(k, self.Wk)
        V = np.dot(v, self.Wv)

        grad['Q_q']  = np.moveaxis(self.Wq, -1, -2)
        grad['Q_Wq'] = np.moveaxis(q, -1, -2)
        grad['K_k']  = np.moveaxis(self.Wk, -1, -2)
        grad['K_Wk'] = np.moveaxis(k, -1, -2)
        grad['V_v']  = np.moveaxis(self.Wv, -1, -2)
        grad['V_Wv'] = np.moveaxis(v, -1, -2)

        Q = np.moveaxis(Q, -2, 0)
        K = np.moveaxis(K, -2, 0)
        V = np.moveaxis(V, -2, 0)

        a = masked_attention(Q, K, V, mask) # (h, N, L_q, d_k)

        concat_heads = np.concatenate(a, axis=-1)
        mha = concat_heads @ self.Wo

        return mha

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class MultiHeadSelfAttention:
    """Similar to `MultiHeadCrossAttention` but assumes the q, k, v
    matrices are the same x.
    
    This leads to two differences which help with computational efficiency:
    
    a) does not allow independent d_v (dimension of the value matrix projection)
    b) packs the Q, K, and V matrices in a larger matrix for multiplication
    """
    
    def __init__(self, d_model, num_heads, seed=None):
        """Expects a batch of shape (N, L, d_model) where
            N: batch size
            L: sequence length
        
        Also handles unbatched input shape (L, d_model)
        """
        
        assert d_model % num_heads == 0, "Model (embedding) dimension must be divisible by the number of heads."

        d_k = d_model // num_heads

        self.qkv_proj_weights = xavier_uniform((num_heads, d_model, 3*d_k), seed=seed)
        self.Wo = xavier_uniform((d_model, d_model), seed=seed)
    
    def forward(self, x, mask=None):
        """x of shape (N, L, d_model)"""

        qkv_proj = np.dot(x, self.qkv_proj_weights) # (N, L, h, 3*d_k)
        qkv_proj = np.moveaxis(qkv_proj, -2, 0)     # (h, N, L, 3*d_k)

        Q, K, V = np.split(qkv_proj, 3, axis=-1)    # (h, N, L, d_k) each

        a = masked_attention(Q, K, V, mask)         # (h, N, L, d_k)

        concat_heads = np.concat(a, axis=-1)        # (N, L, d_model)
        mha = concat_heads @ self.Wo                # (N, L, d_model)

        return mha

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


if __name__ == "__main__":
    d_model = 512
    h = 8
    d_v = 32

    L = 10
    N = 16

    kdim, vdim = 96, 112

    random_seed = 0

    rng = np.random.default_rng(seed=random_seed)

    x = rng.standard_normal((N, L, d_model))

    # mhsa = MultiHeadSelfAttention(d_model, h, seed=random_seed)
    # attention = mhsa(x)
    # attention

    ###

    q = rng.standard_normal((N, L, d_model))
    k = rng.standard_normal((N, L, kdim))
    v = rng.standard_normal((N, L, vdim))

    mhca = MultiHeadAttention(d_model, h, kdim=kdim, vdim=vdim, seed=random_seed)
    attention = mhca(q, k, v)
    attention
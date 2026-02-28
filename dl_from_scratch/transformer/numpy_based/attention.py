import numpy as np
from dl_from_scratch.utils.helpers import xavier_uniform, masked_softmax


def scaled_dot_product(Q, K, return_grad=False):

    d_k = Q.shape[-1]
    s = np.sqrt(1./d_k)

    K_T = np.moveaxis(K, -1, -2)

    S = (Q @ K_T) * s # (N, L_q, L_k)

    def grad(upstream_grad):
        return {
            'Q': upstream_grad @ K * s if upstream_grad is not None else K * s,
            'K': np.moveaxis(upstream_grad, -1, -2) @ Q * s if upstream_grad is not None else Q * s,
        }

    return (S, grad) if return_grad else S

def masked_attention(Q, K, V, mask, return_grad=False):

    # computation graph:
    # q --> S --> A ----> a
    # k ----^          /
    # v --------------/

    S, grad_S = scaled_dot_product(Q, K, return_grad=True)
    A, grad_A = masked_softmax(S, mask, axis=-1, return_grad=True)
    a = A @ V # (N, L_k, d_k)

    def grad_fn(upstream_grad):

        local_gradient = {
            'A': np.moveaxis(V, -1, -2),
            'V': np.moveaxis(A, -1, -2),
        }
        
        dLoss_dV = local_gradient['V'] @ upstream_grad
        dLoss_dA = upstream_grad @ local_gradient['A']
        dLoss_dqk = grad_S(grad_A(dLoss_dA))

        dLoss_dInputs = {
            'Q': dLoss_dqk['Q'],
            'K': dLoss_dqk['K'],
            'V': dLoss_dV,
        }

        return dLoss_dInputs
    
    return (a, grad_fn) if return_grad else a

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
    
    def forward(self, q, k, v, mask=None, return_grad=False):
        """q, k, and v all of shape (N, L, d_model)"""

        Q = np.dot(q, self.Wq)
        K = np.dot(k, self.Wk)
        V = np.dot(v, self.Wv)

        Q = np.moveaxis(Q, -2, 0)
        K = np.moveaxis(K, -2, 0)
        V = np.moveaxis(V, -2, 0)

        a, grad_attention = masked_attention(Q, K, V, mask, return_grad=True) # (h, N, L_q, d_k)
        
        concat_heads = np.concatenate(a, axis=-1)
        mha = concat_heads @ self.Wo

        def grad(upstream_grad):
            num_heads = a.shape[0]

            local_grad = {}
            local_grad['Q_q']  = np.moveaxis(self.Wq, -1, -2)
            local_grad['Q_Wq'] = np.moveaxis(q, -1, -2)
            local_grad['K_k']  = np.moveaxis(self.Wk, -1, -2)
            local_grad['K_Wk'] = np.moveaxis(k, -1, -2)
            local_grad['V_v']  = np.moveaxis(self.Wv, -1, -2)
            local_grad['V_Wv'] = np.moveaxis(v, -1, -2)

            local_grad['Wo'] = concat_heads

            local_grad['concat_heads'] = upstream_grad @ self.Wo
            local_grad['a'] = np.stack(np.split(local_grad['concat_heads'], num_heads, axis=-1))
            attn_grad = grad_attention(local_grad['a'])
            
            return {
                'q' : np.sum(attn_grad['Q'], axis=1) @ local_grad['Q_q'],
                'k' : np.sum(attn_grad['K'], axis=1) @ local_grad['K_k'],
                'v' : np.sum(attn_grad['V'], axis=1) @ local_grad['V_v'],
                'Wq': np.sum(local_grad['Q_Wq'], axis=0) @ np.sum(attn_grad['Q'], axis=1),
                'Wk': np.sum(local_grad['K_Wk'], axis=0) @ np.sum(attn_grad['K'], axis=1),
                'Wv': np.sum(local_grad['V_Wv'], axis=0) @ np.sum(attn_grad['V'], axis=1),
                'Wo': np.sum(np.moveaxis(upstream_grad, -1, -2) @ local_grad['Wo'], axis=0), # divide by batch here?
            }

        
        return (mha, grad) if return_grad else mha

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
    k = rng.standard_normal((N, L*2, kdim))
    v = rng.standard_normal((N, L*2, vdim))

    mhca = MultiHeadAttention(d_model, h, kdim=kdim, vdim=vdim, seed=random_seed)
    attention, attn_grad_fn = mhca(q, k, v, return_grad=True)
    attention
import torch
import torch.nn.functional as F


class MultiHeadSelfAttentionMinimal(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.d_k       = d_model // num_heads
        self.num_heads = num_heads
        self.dot_prod_scaling_factor = self.d_k ** 0.5

        self.projection = torch.nn.Parameter(torch.empty(d_model, 3 * d_model)) # (d_model, 3 * d_model)
        self.W_O        = torch.nn.Parameter(torch.empty(d_model, d_model))     # (h * d_v, d_model)
        self.dropout    = torch.nn.Dropout(dropout)

        self._reset_parameters()
    
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection)
        torch.nn.init.xavier_uniform_(self.W_O)
    
    def forward(self, x):
        """x: (b, L, d_model)"""

        Q, KT, V = self._project_inputs(x)
        A = self._scaled_dot_product_attention(Q, KT)
        A = self.dropout(A)
        Z = A @ V # (b, h, L, d_v) -- Attention-weighted values
        out = self._attention_output(Z)
        out = self.dropout(out)

        return out, A
    
    def _project_inputs(self, x):

        Q, K, V = torch.chunk(x @ self.projection, 3, dim=-1)

        Q  = torch.reshape (Q, (*Q.shape[:-1], self.num_heads, -1)) # (b, L, h, d_k)
        Q  = torch.moveaxis(Q, -3, -2) # (b, h, L, d_k)
        
        K  = torch.reshape (K, (*K.shape[:-1], self.num_heads, -1)) # (b, L, h, d_k)
        KT = torch.moveaxis(K, -3, -1) # (b, h, d_k, L)

        V  = torch.reshape (V, (*V.shape[:-1], self.num_heads, -1)) # (b, L_k, h, d_v)
        V  = torch.moveaxis(V, -3, -2)   # (b, h, L_k, d_v)
        
        return Q, KT, V
    
    def _scaled_dot_product_attention(self, Q, KT):

        scaled_dot_product = Q @ KT / self.dot_prod_scaling_factor  # (b, h, L, L)
        masked_logits = self._apply_causal_mask(scaled_dot_product) # (b, h, L, L)

        return F.softmax(masked_logits, dim=-1)                    # (b, h, L, L)

    def _attention_output(self, Z):

        ZT = Z.transpose(-2, -3)                # (b, L, h, d_k)
        Zo = torch.reshape(ZT,
                (*ZT.shape[:-2], self.d_model)) # (b, L, d_model)
        out = Zo @ self.W_O                     # (b, L, d_model)
        
        return out

    def _apply_causal_mask(self, scaled_dot_product):
        mask = torch.triu(torch.full_like(scaled_dot_product, -torch.inf), diagonal=1)
        return scaled_dot_product + mask

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, h=8, d_v=None, kdim=None, vdim=None, dropout=0.1, is_causal=False):
        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.d_k = d_model // h
        self.num_heads = h
        self.d_v = d_v if d_v is not None else self.d_k
        self.dot_prod_scaling_factor = self.d_k ** 0.5
        self.kdim = d_model if kdim is None else kdim
        self.vdim = d_model if vdim is None else vdim
        self.is_causal = is_causal

        self.W_Q = torch.nn.Parameter(torch.empty(d_model  , h * self.d_k))  # (d_model, d_model)
        self.W_K = torch.nn.Parameter(torch.empty(self.kdim, h * self.d_k))  # (kdim   , d_model)
        self.W_V = torch.nn.Parameter(torch.empty(self.vdim, h * self.d_v))  # (vdim   , h * d_v)
        self.W_O = torch.nn.Parameter(torch.empty(h * self.d_v, d_model))    # (h * d_v, d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self._reset_parameters()
    
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)
    
    def forward(self, q, k=None, v=None, apply_causal_mask=None, need_weights=True, average_attn_weights=True):
        """q: (b, L_q, d_model), k: (b, L_k, d_model), v: (b, L_k, d_model)"""

        assert (k is None and v is None) or (k is not None and v is not None), "`k` and `v` should either both be given or both be omitted at the same time."

        if k is None:
            k = v = q
        
        if apply_causal_mask is None:
            apply_causal_mask = self.is_causal

        Q, KT, V = self._project_inputs(q, k, v)
        A = self._scaled_dot_product_attention(Q, KT, apply_causal_mask)
        A = self.dropout(A)
        Z = A @ V # (b, h, L_q, d_v) -- Attention-weighted values
        out = self._attention_output(Z)
        out = self.dropout(out)

        if average_attn_weights:
            A = torch.mean(A, dim=1)

        return (out, A) if need_weights else out
    
    def _project_inputs(self, q, k, v):
        
        if self.d_model == self.kdim == self.vdim and self.d_k == self.d_v and k is v:
            if q is k:
                proj_weights = torch.concatenate((self.W_Q, self.W_K, self.W_V), dim=-1)
                Q, K, V = torch.chunk(q @ proj_weights, 3, dim=-1)
            else:
                Q = q @ self.W_Q
                proj_weights = torch.concatenate((self.W_K, self.W_V), dim=-1)
                K, V = torch.chunk(k @ proj_weights, 2, dim=-1)
        else:
            Q = q @ self.W_Q # (b, L_q, h * d_k)
            K = k @ self.W_K # (b, L_k, h * d_k)
            V = v @ self.W_V # (b, L_k, h * d_v)
        
        Q  = torch.reshape(Q, (*Q.shape[:-1], self.num_heads, -1)) # (b, L_q, h, d_k)
        K  = torch.reshape(K, (*K.shape[:-1], self.num_heads, -1)) # (b, L_k, h, d_k)
        Q  = torch.moveaxis(Q, -3, -2) # (b, h, L_q, d_k)
        KT = torch.moveaxis(K, -3, -1) # (b, h, d_k, L_k)

        V  = torch.reshape(V,
                    (*V.shape[:-1], self.num_heads, -1)
                    )    # (b, L_k, h, d_v)
        V  = torch.moveaxis(
            V, -3, -2)   # (b, h, L_k, d_v)
        
        return Q, KT, V
    
    def _scaled_dot_product_attention(self, Q, KT, causal_mask=False):

        scaled_dot_product = Q @ KT / self.dot_prod_scaling_factor     # (b, h, L_q, L_k)
        masked_logits = self._apply_mask_if_causal(scaled_dot_product, causal_mask) # (b, h, L_q, L_k)
        
        A  = F.softmax(masked_logits, dim=-1) # (b, h, L_q, L_k)

        return A

    def _attention_output(self, Z):

        ZT = Z.transpose(-2, -3) # (b, L_q, h, d_v)
        Zo = torch.reshape(ZT,
                    (*ZT.shape[:-2], self.num_heads * self.d_v)
                    )            # (b, L_q, h * d_v)
        out = Zo @ self.W_O      # (b, L_q, d_model)
        
        return out

    def _apply_mask_if_causal(self, scaled_dot_product, causal_mask=False):

        if causal_mask:
            mask = torch.triu(torch.full_like(scaled_dot_product, -torch.inf), diagonal=1)
        else:
            mask = torch.zeros_like(scaled_dot_product)
        
        return scaled_dot_product + mask

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

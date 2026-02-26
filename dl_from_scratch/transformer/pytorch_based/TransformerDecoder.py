import torch
from torch import nn
import torch.nn.functional as F

from Attention import MultiHeadAttention

class DecoderOnlyLayer(torch.nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, is_causal=False):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        
        self.LN1 = nn.LayerNorm(d_model)
        self.MHA = MultiHeadAttention(d_model, h, is_causal=is_causal)
        self.LN2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, causal=False):

        x = self.LN1(x)
        x = x + self.MHA.forward(x, apply_causal_mask=causal, need_weights=False)

        x = self.LN2(x)
        x = x + self.MLP(x)

        return x

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

class DecoderOnlyTransformer(torch.nn.Module):
    def __init__(self, dict_size, d_model=512, num_heads=8, n_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.h = num_heads

        self.embedding = torch.nn.Embedding(dict_size, d_model)
        with torch.no_grad():
            nn.init.xavier_normal_(self.embedding.weight)

        self.Layers = torch.nn.Sequential(
            *(DecoderOnlyLayer(is_causal=True) for _ in range(n_layers)),
        )
    
    def forward(self, x):
        embedding = self.embedding(x)
        processed = self.Layers(embedding)
        logits = processed @ self.embedding.weight.T # weight sharing, as in §3.4 of the paper
        return F.softmax(logits, dim=-1)

if __name__ == "__main__":
    h = 8
    N, L_q, L_k, d_model, d_v = 16, 12, 10, 512, 64
    kdim, vdim = 112, 148

    N, L_q, L_k, d_model, d_v = 16, 12, 10, 512, 64
    kdim, vdim = 512, 512

    dict_size = 10_000
    tokens = torch.randint(0, dict_size, (N, L_q,))

    mha = MultiHeadAttention(d_model=d_model, d_v=d_v, kdim=kdim, vdim=vdim, dropout=0.1, is_causal=False)
    mhsa = MultiHeadSelfAttentionMinimal(d_model=d_model, dropout=0.1)
    mha_torch = nn.MultiheadAttention(d_model, num_heads=h, dropout=0.1, kdim=kdim, vdim=vdim, bias=False, batch_first=True)

    if mha_torch.in_proj_weight is not None:
        wq, wk, wv = torch.chunk(mha_torch.in_proj_weight, 3)
    else:
        wq = mha_torch.q_proj_weight
        wk = mha_torch.k_proj_weight
        wv = mha_torch.v_proj_weight
    wo = mha_torch.out_proj.weight

    with torch.no_grad():
        mha.W_Q[:] = wq.T
        mha.W_K[:] = wk.T
        mha.W_V[:] = wv.T
        mha.W_O[:] = wo.T
        
        mhsa.projection[:] = mha_torch.in_proj_weight.T
        mhsa.W_O[:]        = wo.T

    q = torch.rand((N, L_q, d_model))
    k = torch.rand((N, L_k, kdim))
    v = torch.rand((N, L_k, vdim))

    mha.eval()
    mhsa.eval()
    mha_torch.eval()

    q_sa, A_sa           = mhsa.forward(q)
    q_new, A             = mha.forward (q, q, q, average_attn_weights=False)
    q_new_torch, A_torch = mha_torch   (q, q, q, average_attn_weights=False)

    print(f"Q: the same? {torch.allclose(q_new_torch, q_new, atol=1e-5)}")
    print(f"A: the same? {torch.allclose(A_torch, A, atol=1e-5)}")
    print()
    print(f"Q_minimal: the same? {torch.allclose(q_new_torch, q_sa, atol=1e-5)}")
    print(f"A_minimal: the same? {torch.allclose(A_torch, A_sa, atol=1e-5)}")
    print()

    tx = DecoderOnlyTransformer(dict_size)

    # predictions = tx(tokens)
    # print(f"transformer preds: {predictions}")

# We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized
import numpy as np

from attention import MultiHeadSelfAttention
from normalizations import LayerNorm
from embeddings import EmbeddingLayer
from dl_from_scratch.utils.helpers import softmax
from dl_from_scratch.mlp.two_layer_nn import TwoLayerMLP, LinearLayer

class DecoderLayer():
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, d_v=None, random_seed=None):
        """
        `d_v` is ignored (for now).
        `random_seed` is for initialization.
        """

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = num_heads
        self.d_v = d_v
        
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, seed=random_seed)
        self.norm2 = LayerNorm(d_model)
        self.MLP = TwoLayerMLP(d_model, d_ff, d_model)
    
    def forward(self, x, mask=None):

        x = self.norm1(x)
        x = x + self.attention(x, mask=mask)

        x = self.norm2(x)
        x = x + self.MLP(x)

        return x

class DecoderOnlyTransformer():
    def __init__(self, dict_size, N_layers=6, d_model=512, num_heads=8, d_ff=2048):
        self.d_model = d_model
        self.h = num_heads

        self.embedding = EmbeddingLayer(dict_size, d_model)

        self.Decoders = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(N_layers)]
        self.Linear = LinearLayer(d_model, dict_size)
    
    def forward(self, x):
        # x: (N, L)

        N, L = x.shape
        causal_mask = np.triu(np.ones((L, L), dtype=bool))

        embedding = self.embedding(x) # (N, L, d_model)

        representation = embedding
        for layer in self.Decoders:
            representation = layer(representation, mask=causal_mask)
        
        logits = self.Linear(representation)
        
        return softmax(logits, axis=-1)

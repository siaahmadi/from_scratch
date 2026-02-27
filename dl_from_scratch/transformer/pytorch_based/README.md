# Transformer implementatino from scratch

This is an implementation of the Transformer from scratch using PyTorch primitives. Currently it only impelments a decoder-only (causal) transformer. It is slightly more flexible than the PyTorch implementation. It tries to stay true to the original paper but has some changes (for example, it uses pre-norm instead of post-norm).
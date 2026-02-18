import numpy as np
import warnings

class TwoLayerMLP:
    def __init__(self, d_in, d_h, d_out, learning_rate, dtype=np.float32):
        kaiming_factor = lambda d_in: np.sqrt(2. / d_in)

        self.W1 = (np.random.standard_normal((d_in, d_h)) * kaiming_factor(d_in)).astype(dtype)
        self.b1 = np.zeros((1, d_h), dtype=dtype)
        self.W2 = (np.random.standard_normal((d_h, d_out)) * kaiming_factor(d_h)).astype(dtype)
        self.b2 = np.zeros((1, d_out), dtype=dtype)

        self.lr = learning_rate

        # cache gradients
        self._grad = {
            "h1_b1": np.ones_like(self.b1), # 1 x H
            "h3_b2": np.ones_like(self.b2), # 1 x C
        }
    
    def forward(self, x):
        # N: num_data, D: dim_data, H: hidden_size, C: num_classes

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
                
            h1 = x @ self.W1 + self.b1    # N x H = (N x D) @ (D x H) + (1 x H)
            h2 = np.maximum(h1, 0)        # N x H   (ReLU)
            h3 = h2 @ self.W2 + self.b2   # N x C = (N x H) @ (H x C) + (1 x C)

            # ∇h_1
            self._grad['h1_W1'] = x.T       # D x N

            # ∇h_2
            self._grad['h2_h1'] = np.ones_like(h1)
            self._grad['h2_h1'][h1 < 0] = 0 # N x H

            # ∇h_3
            self._grad['h3_W2'] = h2.T      # H x N
            self._grad['h3_h2'] = self.W2.T # C x H

            return h3

    def backward(self, dL_dh3):
        N = dL_dh3.shape[0]
        assert dL_dh3.shape == (N, self.W2.shape[1]) # N x C

        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            self._grad['L_h3'] = dL_dh3 / N                               # N x C
            
            self._grad['L_W2'] = self._grad['h3_W2'] @ self._grad['L_h3'] # H x C = (H x N) @ (N x C)

            self._grad['L_b2'] = self._grad['h3_b2'] * self._grad['L_h3'] # N x C = (1 x C) * (N x C)
            self._grad['L_b2'] = np.sum(self._grad['L_b2'],
                                        axis=0, keepdims=True)            # 1 x C

            self._grad['L_h2'] = self._grad['L_h3'] @ self._grad['h3_h2'] # N x H = (N x C) @ (C x H)

            self._grad['L_h1'] = self._grad['L_h2'] * self._grad['h2_h1'] # N x H

            self._grad['L_W1'] = self._grad['h1_W1'] @ self._grad['L_h1'] # D x H = (D x N) @ (N x H)

            self._grad['L_b1'] = self._grad['h1_b1'] * self._grad['L_h1'] # N x H = (1 x H) * (N x H)
            self._grad['L_b1'] = np.sum(self._grad['L_b1'],
                                        axis=0, keepdims=True)            # 1 x H

    def step(self):
        self.W1 += - self._grad['L_W1'] * self.lr
        self.b1 += - self._grad['L_b1'] * self.lr
        self.W2 += - self._grad['L_W2'] * self.lr
        self.b2 += - self._grad['L_b2'] * self.lr
    
    def gradient_check(self, x):
        pass # todo

import numpy as np

np.random.seed(42)

class Dropout:
    def __init__(self, keep_prob = 0.5):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X, training=True):
        if not training or self.keep_prob == 1.0:
            return X
        self.mask = (np.random.rand(*X.shape) < self.keep_prob) / self.keep_prob
        return X * self.mask
        
    def backward(self, dX):
        if self.mask is None:
            return dX
        return dX * self.mask
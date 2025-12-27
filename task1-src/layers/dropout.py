import numpy as np

np.random.seed(42)

class Dropout:
    '''Implements inverted dropout regularisation'''
    def __init__(self, keep_prob = 0.5):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X, training=True):
        # During inference or when dropout is disabled, return input unchanged
        if not training or self.keep_prob == 1.0:
            return X
        
        # Generate dropout mask and scale activations (inverted dropout)
        self.mask = (np.random.rand(*X.shape) < self.keep_prob) / self.keep_prob
        return X * self.mask
        
    def backward(self, dX):
        # Propagate gradients only through active neurons
        if self.mask is None:
            return dX
        return dX * self.mask
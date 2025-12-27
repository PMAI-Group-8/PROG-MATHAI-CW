import numpy as np

np.random.seed(42)

class L2Regularizer:
    '''L2 regularisation to penalise large weights'''

    def __init__(self, lam = 1e-4):
        # Regularisation strength
        self.lam = lam

    
    def penalty(self, W):
        # Computes the L2 penalty term added to the loss
        return self.lam * np.sum(W ** 2)
    
    def gradient(self, W):
        # Gradient of the L2 penalty with respect to weights
        return self.lam * W
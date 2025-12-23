import numpy as np

np.random.seed(42)

class L2Regularizer:
    def __init__(self, lam = 1e-4):
        self.lam = lam

    
    def penalty(self, W):
        return self.lam * np.sum(W ** 2)
    
    def gradient(self, W):
        return self.lam * W
    
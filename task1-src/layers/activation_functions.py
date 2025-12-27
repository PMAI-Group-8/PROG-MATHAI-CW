import numpy as np

# Fix random seed for reproducibility
np.random.seed(42)

class activation():
    """
    Base activation class.
    All activation functions inherit from this and
    implement their own forward and backward passes.
    """
    def forward(self, Z):
        raise NotImplementedError("Forward method not implemented.")
    
    def backward(self, dA):
        raise NotImplementedError("Backward method not implemented.")

class ReLU(activation):
    '''ReLU Activation Function'''
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        return dA * (self.Z > 0).astype(float)
    
class Sigmoid(activation):
    '''Sigmoid Activation Function'''
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1 - self.A)
import numpy as np

np.random.seed(42)

class NeuronLayer:
    def __init__(self, n_inputs, n_outputs, 
        activation = None, dropout = None, 
        l2 = None):
        
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.b = np.zeros((1, n_outputs))
        self.activation = activation
        self.dropout = dropout
        self.l2 = l2
    
    ''' Forward pass through the layer '''
    def forward(self, X, training=True):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b

        A = self.Z
        if self.activation:
            A = self.activation.forward(self.Z)
        if self.dropout:
            A = self.dropout.forward(A, training)
        return A
    
    ''' Backward pass through the layer '''
    def backward(self, dA):
        if self.dropout:
            dA = self.dropout.backward(dA)
        dZ = dA
        if self.activation:
            dZ = self.activation.backward(dA)

        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)

        if self.l2 is not None:
            dW += self.l2.gradient(self.W)
        return dX, dW, db
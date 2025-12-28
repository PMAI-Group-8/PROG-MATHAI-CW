import numpy as np

np.random.seed(42)

class NeuronLayer:
    def __init__(self, n_inputs, n_outputs, 
        activation=None, dropout=None, 
        l2=None, layer_id=None):
        
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_outputs))
        self.activation = activation
        self.dropout = dropout
        self.l2 = l2
        self.layer_id = layer_id
    
    ''' Forward pass through the layer '''
    def forward(self, X, training=True):
        self.X = X
        self.Z = X @ self.W + self.b 

        A = self.Z
        if self.activation:
            A = self.activation.forward(self.Z)
        if self.dropout:
            A = self.dropout.forward(A, training)

        self.A = A
        return A

    
    ''' Backward pass through the layer '''
    def backward(self, dA):
        # 1. Dropout backward
        if self.dropout:
            dA = self.dropout.backward(dA)

        # 2. Activation backward
        dZ = self.activation.backward(dA) if self.activation else dA

        m = self.X.shape[0]
        inv_m = 1.0 / m

        dW = (self.X.T @ dZ) * inv_m 
        db = np.sum(dZ, axis=0, keepdims=True) * inv_m
        dX = dZ @ self.W.T

        if self.l2 is not None:
            dW += self.l2.gradient(self.W)

        return dX, dW, db
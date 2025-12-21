import numpy as np

np.random.seed(42)

class NeuronLayer:
    def __init__(self, n_inputs, n_outputs, activation = None):
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.b = np.zeros((1, n_outputs))
        self.activation = activation
    
    ''' Forward pass through the layer '''
    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b

        A = self.Z
        if self.activation:
            A = self.activation.forward(self.Z)
        return A
    
    ''' Backward pass through the layer '''
    def backward(self, dA, learning_rate):
        dZ = dA
        if self.activation:
            dZ = self.activation.backward(dA)

        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)

        ''' Update weights and biases with stochastic gradient descent '''
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX
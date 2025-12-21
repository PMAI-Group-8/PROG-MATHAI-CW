import numpy as np

np.random.seed(42)

from layers.loss import SoftmaxCrossEntropy
from neuron_layer import NeuronLayer
from layers.activation_functions import ReLU, Sigmoid

class NeuralNetwork:
    def __init__(self):
        self.layer1 = NeuronLayer(n_inputs=3072, n_outputs=128, activation=ReLU())
        self.layer2 = NeuronLayer(n_inputs=128, n_outputs=64, activation=ReLU())
        self.layer3 = NeuronLayer(n_inputs=64, n_outputs=10, activation=None)
        self.loss_function = SoftmaxCrossEntropy()

    ''' Forward pass through the network '''
    ''' Result is not one-hot encoded; softmax is applied in the loss function '''
    def forward(self, X):
        A1 = self.layer1.forward(X)
        A2 = self.layer2.forward(A1)
        A3 = self.layer3.forward(A2)
        return A3
    
    ''' Backward pass through the network '''
    def backward(self, dLoss):
        dA3 = self.layer3.backward(dLoss, learning_rate=0.01)
        dA2 = self.layer2.backward(dA3, learning_rate=0.01)
        dA1 = self.layer1.backward(dA2, learning_rate=0.01)
        return dA1
    
    ''' Compute loss and its gradient '''
    def compute_loss_and_gradient(self, Y_pred, Y_true):
        loss = self.loss_function.forward(Y_pred, Y_true)
        dLoss = self.loss_function.backward()
        return loss, dLoss
    
    ''' One-hot encode labels '''
    def one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    ''' Train the neural network '''
    def train(self, X, Y_true, epochs = 100):
        y_onehot = self.one_hot_encode(Y_true, num_classes=10)
        for epoch in range(epochs):
            Y_pred = self.forward(X)

            loss, dLoss = self.compute_loss_and_gradient(Y_pred, y_onehot)

            self.backward(dLoss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")



import numpy as np

np.random.seed(42)

from layers.loss import SoftmaxCrossEntropy
from neuron_layer import NeuronLayer
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout

class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layer1 = NeuronLayer(
            n_inputs=3072, 
            n_outputs=128, 
            activation=ReLU(), 
            dropout=Dropout(keep_prob=0.8)
        )
        
        self.layer2 = NeuronLayer(
            n_inputs=128, 
            n_outputs=64, 
            activation=ReLU(), 
            dropout=Dropout(keep_prob=0.8)
        )
        
        self.layer3 = NeuronLayer(
            n_inputs=64, 
            n_outputs=10, 
            activation=None,
            dropout=None
        )
        self.learning_rate = learning_rate
        self.loss_function = SoftmaxCrossEntropy()

    ''' Forward pass through the network '''
    ''' Result is not one-hot encoded; softmax is applied in the loss function '''
    def forward(self, X, training = True):
        A1 = self.layer1.forward(X, training=training)
        A2 = self.layer2.forward(A1, training=training)
        A3 = self.layer3.forward(A2, training=training)
        return A3
    
    ''' Backward pass through the network '''
    def backward(self, dLoss):
        dA3 = self.layer3.backward(dLoss, self.learning_rate)
        dA2 = self.layer2.backward(dA3, self.learning_rate)
        dA1 = self.layer1.backward(dA2, self.learning_rate)
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
            Y_pred = self.forward(X, training=True)

            loss, dLoss = self.compute_loss_and_gradient(Y_pred, y_onehot)

            self.backward(dLoss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    '''Predict class labels for given input'''
    def predict(self, X):
        Y_pred = self.forward(X, training=False)
        predictions = np.argmax(Y_pred, axis=1)
        return predictions
    
    ''' Calculate overall accuracy '''
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)




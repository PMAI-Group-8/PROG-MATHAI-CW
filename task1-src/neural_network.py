import numpy as np

np.random.seed(42)

from layers.loss import SoftmaxCrossEntropy
from neuron_layer import NeuronLayer
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout

'''
EEEDWARD1 implemented:
- preprocessing pipeline in preprocessor.py
- 
'''

class NeuralNetwork:
    def __init__(self, layer_config, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.loss_function = SoftmaxCrossEntropy()

        for cfg in layer_config:
            self.layers.append(NeuronLayer(
                cfg['n_inputs'],
                cfg['n_neurons'],
                cfg.get('activation'),
                cfg.get('dropout'),
                )
            )

    ''' Forward pass through the network '''
    ''' Result is not one-hot encoded; softmax is applied in the loss function '''
    def forward(self, X, training = True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X
    
    ''' Backward pass through the network '''
    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss, self.learning_rate)
        return dLoss
    
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
    def train(self, X, Y_true, X_val, y_val, epochs = 100):
        y_onehot = self.one_hot_encode(Y_true, num_classes=10)
        for epoch in range(epochs):
            Y_pred = self.forward(X, training=True)

            loss, dLoss = self.compute_loss_and_gradient(Y_pred, y_onehot)

            self.backward(dLoss)

            if epoch % 10 == 0:
                val_accuracy = self.accuracy(X_val, y_val)
                train_accuracy = self.accuracy(X, Y_true)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.2f}, Validation Accuracy: {val_accuracy:.2f}")
                
    '''Predict class labels for given input'''
    def predict(self, X):
        Y_pred = self.forward(X, training=False)
        predictions = np.argmax(Y_pred, axis=1)
        return predictions
    
    ''' Calculate overall accuracy '''
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)




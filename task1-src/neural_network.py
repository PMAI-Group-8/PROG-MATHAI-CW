import numpy as np
 
np.random.seed(42)
 
from layers.loss import SoftmaxCrossEntropy
from neuron_layer import NeuronLayer
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout
from layers.optimisers import SGD, SGDWithMomentum, Adam
 
class NeuralNetwork:
    def __init__(self, layer_config, optimiser=None):
        self.layers = []
        self.optimiser = optimiser if optimiser else SGD(learning_rate=0.01)
        self.loss_function = SoftmaxCrossEntropy()
 
        for i,cfg in enumerate(layer_config):
            self.layers.append(NeuronLayer(
                cfg['n_inputs'],
                cfg['n_neurons'],
                cfg.get('activation'),
                cfg.get('dropout'),
                cfg.get('l2'),
                layer_id=i
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
            dLoss, dW, db = layer.backward(dLoss)
            if self.optimiser:
                self.optimiser.update(layer, dW, db)
        return dLoss
    
    ''' Compute loss and its gradient '''
    def compute_loss_and_gradient(self, Y_pred, Y_true):
        data_loss = self.loss_function.forward(Y_pred, Y_true)
        dLoss = self.loss_function.backward()
 
        total_loss = data_loss + self.l2_loss()
        return total_loss, dLoss
    
    
    ''' One-hot encode labels '''
    def one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    ''' Train the neural network '''
    def train(self, X, Y_true, X_val, y_val, epochs = 100, batch_size=32):
        y_onehot = self.one_hot_encode(Y_true, num_classes=10)
        n = X.shape[0]
 
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_onehot_shuffled = y_onehot[idx]
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_onehot_batch = y_onehot_shuffled[i:i+batch_size]
 
                Y_pred = self.forward(X_batch, training=True)
 
                loss, dLoss = self.compute_loss_and_gradient(Y_pred, y_onehot_batch)
 
                self.backward(dLoss)
 
            if epoch % 10 == 0 or epoch == epochs - 1:
                train_accuracy = self.accuracy(X, Y_true)
                val_accuracy = self.accuracy(X_val, y_val)
                print(
                    f"Epoch {epoch}, "
                    f"Loss: {loss:.4f}, "
                    f"Training Accuracy: {train_accuracy:.2f}, "
                    f"Validation Accuracy: {val_accuracy:.2f}"
                )
                
    '''Predict class labels for given input'''
    def predict(self, X):
        Y_pred = self.forward(X, training=False)
        predictions = np.argmax(Y_pred, axis=1)
        return predictions
    
    ''' Calculate overall accuracy '''
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
    
    def l2_loss(self):
        loss = 0.0
        for layer in self.layers:
            if layer.l2 is not None:
                loss += layer.l2.penalty(layer.W)
        return loss

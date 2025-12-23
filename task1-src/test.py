# %%
from Cifar10Loader import Cifar10Loader
from neural_network import NeuralNetwork
from neuron_layer import NeuronLayer
from preprocessor import Preprocessor, PreprocessorAdvanced
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout

import numpy as np
np.random.seed(42)

# %%
CIFAR_DIR = ".\dataset\cifar10"

loader = Cifar10Loader(CIFAR_DIR)

X_train_raw, y_train = loader.load_train_data()
X_test_raw, y_test = loader.load_test_data()

print(X_train_raw.shape, y_train.shape)

# %%
preprocessor = PreprocessorAdvanced()
X_train, y_train, X_val, y_val = preprocessor.preprocess_train_data(X_train_raw, y_train)
X_test, y_test = preprocessor.preprocess_test_data(X_test_raw, y_test)

# %%
print('Training: ', X_train.shape, y_train.shape)
print('Validation: ', X_val.shape, y_val.shape)
print('Test: ', X_test.shape, y_test.shape)

print(X_train.mean(), X_train.std())

# %%
layer_config = [
    {
        'n_inputs': 3072, 
        'n_neurons': 1024, 
        'activation': ReLU(), 
        'dropout': Dropout(0.7),
        'l2' : 0.001
    },
    {
        'n_inputs': 1024,  
        'n_neurons': 256, 
        'activation': ReLU(), 
        'dropout': Dropout(0.9)
    },
    {
        'n_inputs': 256,  
        'n_neurons': 10,  
        'activation': None
    }
]

# %%
### Test 1
# Learning rate 0.01
nn = NeuralNetwork(layer_config, learning_rate=0.1)
nn.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)

# %%
accuracy = nn.accuracy(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")


# %%

CIFAR_DIR = ".\dataset\cifar10"

loader = Cifar10Loader(CIFAR_DIR)

X_train_raw, y_train = loader.load_train_data()
X_test_raw, y_test = loader.load_test_data()

print(X_train_raw.shape, y_train.shape)

preprocessor = Preprocessor()
X_train, y_train, X_val, y_val = preprocessor.preprocess_train_data(X_train_raw, y_train)
X_test, y_test = preprocessor.preprocess_test_data(X_test_raw, y_test)
print('Training: ', X_train.shape, y_train.shape)
print('Validation: ', X_val.shape, y_val.shape)
print('Test: ', X_test.shape, y_test.shape)

layer_config = [
    {
        'n_inputs': 3072, 
        'n_neurons': 1024, 
        'activation': ReLU(), 
        'dropout': Dropout(0.7)
    },
    {
        'n_inputs': 1024,  
        'n_neurons': 256, 
        'activation': ReLU(), 
        'dropout': Dropout(0.9)
    },
    {
        'n_inputs': 256,  
        'n_neurons': 10,  
        'activation': None
    }
]

nn = NeuralNetwork(layer_config, learning_rate=0.1)
nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=128)

accuracy = nn.accuracy(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# %%

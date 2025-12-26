from Cifar10Loader import Cifar10Loader
from neural_network import NeuralNetwork
from neuron_layer import NeuronLayer
from preprocessor import Preprocessor, PreprocessorAdvanced
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout
from layers.optimisers import SGD, SGDWithMomentum, Adam
from datacatcher import DataCatcher
from layers.l2_regulariser import L2Regularizer


import numpy as np

np.random.seed(42)

CIFAR_DIR = "./dataset/cifar10"
BASE_DIR = "./task1-src/logs/"
EPOCHS = 101
BATCH_SIZE = 256

loader = Cifar10Loader(CIFAR_DIR)
X_train_raw, y_train_raw = loader.load_train_data()
X_test_raw, y_test_raw = loader.load_test_data()

preprocessor = Preprocessor()
X_train, y_train, X_val, y_val = preprocessor.preprocess_train_data(
    X_train_raw, y_train_raw, train_ratio=0.9
)
X_test, y_test = preprocessor.preprocess_test_data(
    X_test_raw, y_test_raw
)

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

layer_config = [
    {
        'n_inputs': 3072,
        'n_neurons': 512,
        'activation': Sigmoid(),
        'dropout': None,
        'l2': None
    },
    {
        'n_inputs': 512,
        'n_neurons': 128,
        'activation': Sigmoid(),
        'dropout': None,
        'l2': None
    },
    {
        'n_inputs': 128,
        'n_neurons': 10,
        'activation': None,
        'dropout': None,
        'l2': None
    }
]

data_catcher = DataCatcher(
    base_dir=BASE_DIR,
    config={
        "experiment_name": "cifar10_sigmoid_512_128_bs256",
        "metrics": True,
        "activation_logging": True,
        "activation_type": "sigmoid",
        "layers": [0, 1]  # only hidden layers
    }
)

network = NeuralNetwork(
    layer_config=layer_config,
    optimiser=SGD(
        learning_rate=0.01,
        decay=0
    )
)

network.train(
    X=X_train,
    Y_true=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    data_catcher=data_catcher
)

data_catcher.save_activation_logs()

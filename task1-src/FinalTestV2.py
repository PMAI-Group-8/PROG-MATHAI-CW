import numpy as np
import time

from Cifar10Loader import Cifar10Loader
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
from datacatcher import DataCatcher

from layers.activation_functions import ReLU, Sigmoid
from layers.optimisers import SGD, SGDWithMomentum, Adam
from layers.l2_regulariser import L2Regularizer

# --------------------------------------------------
# Global settings
# --------------------------------------------------
np.random.seed(42)

CIFAR_DIR = "./dataset/cifar10"
LOG_DIR = "./task1-src/logs"

# --------------------------------------------------
# Experiment definitions
# --------------------------------------------------
EXPERIMENTS = [
    {
        "name": "T1b_relu_sgd_bs256_epochs101",
        "activation": "relu",
        "batch_size": 256,
        "epochs": 101,
        "optimiser": "sgd",
        "learning_rate": 0.1,
        "momentum": None,
        "l2": None
    },
    {
        "name": "T1b_sigmoid_sgd_bs256_epochs101",
        "activation": "sigmoid",
        "batch_size": 256,
        "epochs": 101,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "momentum": None,
        "l2": None
    },
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_optimiser(cfg):
    if cfg["optimiser"] == "sgd":
        return SGD(learning_rate=cfg["learning_rate"])
    if cfg["optimiser"] == "momentum":
        return SGDWithMomentum(
            learning_rate=cfg["learning_rate"],
            momentum=cfg["momentum"]
        )
    if cfg["optimiser"] == "adam":
        return Adam(learning_rate=cfg["learning_rate"])
    raise ValueError("Unknown optimiser")

def build_layers(activation_name, l2):
    """
    IMPORTANT:
    Each layer gets its OWN activation instance.
    """
    if activation_name == "relu":
        act1 = ReLU()
        act2 = ReLU()
    elif activation_name == "sigmoid":
        act1 = Sigmoid()
        act2 = Sigmoid()
    else:
        raise ValueError("Unknown activation")

    return [
        {
            "n_inputs": 3072,
            "n_neurons": 1024,
            "activation": act1,
            "dropout": None,
            "l2": l2
        },
        {
            "n_inputs": 1024,
            "n_neurons": 256,
            "activation": act2,
            "dropout": None,
            "l2": l2
        },
        {
            "n_inputs": 256,
            "n_neurons": 10,
            "activation": None,
            "dropout": None,
            "l2": None
        }
    ]

# --------------------------------------------------
# Load data 
# --------------------------------------------------
print("Loading CIFAR-10...")
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

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# --------------------------------------------------
# Run experiments
# --------------------------------------------------
for cfg in EXPERIMENTS:
    print("\n" + "=" * 60)
    print(f"Running experiment: {cfg['name']}")
    print("=" * 60)

    # Stabilise sigmoid runs
    if cfg["activation"] == "sigmoid":
        np.random.seed(42)

    l2 = L2Regularizer(cfg["l2"]) if cfg["l2"] else None

    layer_config = build_layers(cfg["activation"], l2)
    optimiser = build_optimiser(cfg)

    # ---- DataCatcher ----
    data_catcher = DataCatcher(
        base_dir=LOG_DIR,
        config={
            "experiment_name": cfg["name"],
            "metrics": True,
            "activation_logging": True,
            "activation_type": cfg["activation"],
            "layers": [0, 1]
        }
    )

    # ---- Network ----
    network = NeuralNetwork(
        layer_config=layer_config,
        optimiser=optimiser
    )

    start = time.time()

    network.train(
        X=X_train,
        Y_true=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        data_catcher=data_catcher
    )

    duration = time.time() - start
    test_acc = network.accuracy(X_test, y_test)

    data_catcher.save_activation_logs()

    print(f"Finished {cfg['name']}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Time taken  : {duration:.2f}s")

print("\nAll experiments completed.")

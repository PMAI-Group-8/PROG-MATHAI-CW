import numpy as np
import time

from Cifar10Loader import Cifar10Loader
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
from datacatcher import DataCatcher

from layers.activation_functions import ReLU, Sigmoid
from layers.optimisers import SGD, SGDWithMomentum, Adam
from layers.l2_regulariser import L2Regularizer
from layers.dropout import Dropout

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
        "name": "T1b_relu_sgd_bs256_epochs150",
        "activation": "relu",
        "activation_logging": True,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.1,
        "decay" : None,
        "momentum": None,
        "dropout": None,
        "l2": None
    },
    {
        "name": "T1b_sigmoid_sgd_bs256_epochs150",
        "activation": "sigmoid",
        "activation_logging": True,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.1,
        "decay" : None,
        "momentum": None,
        "dropout": None,
        "l2": None
    },
    {
        "name": "T1d_relu_sgd_no_dropout",
        "activation": "relu",
        "activation_logging": True,
        "batch_size": 256,
        "epochs": 100,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay" : None,
        "momentum": None,
        "dropout": None,
        "l2": None
    },
    {
        "name": "T1d_relu_sgd_dropout0.5",
        "activation": "relu",
        "activation_logging": True,
        "batch_size": 256,
        "epochs": 100,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay" : None,
        "momentum": None,
        "dropout": 0.5,
        "l2": None
    },
    {
        "name": "T1f_relu_sgd_momentum_0.8_l2_0.001",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay" : None,
        "momentum": 0.8,
        "dropout": None,
        "l2": 0.001
    },
    {
        "name": "T1f_relu_adam_l2_0.001",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "adam",
        "learning_rate": 0.05,
        "decay" : None,
        "momentum": None,
        "dropout": None,
        "l2": 0.001
    },
    {
        "name": "T1g_relu_sgd_momentum_0.9",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 120,
        "optimiser": "momentum",
        "learning_rate": 0.05,
        "decay": None,
        "momentum": 0.9,
        "dropout": None,
        "l2": None
    },
    {
        "name": "T1g_relu_adam_lr_0.001",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 100,
        "optimiser": "adam",
        "learning_rate": 0.001,
        "decay": None,
        "momentum": None,
        "dropout": None,
        "l2": None
    },
    {
        "name": "T1g_relu_dropout0.8",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay": 1e-3,
        "momentum": None,
        "dropout": 0.8,
        "l2": None
    },
    {
        "name": "T1g_relu_l2_0.0005",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay": 1e-3,
        "momentum": None,
        "dropout": None,
        "l2": 0.0005
    },
    {
        "name": "T1g_relu_dropout0.3_l2_0.0005",
        "activation": "relu",
        "activation_logging": False,
        "batch_size": 256,
        "epochs": 150,
        "optimiser": "sgd",
        "learning_rate": 0.05,
        "decay": 1e-3,
        "momentum": None,
        "dropout": 0.8,
        "l2": 0.0005
    }
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_optimiser(cfg):
    if cfg["optimiser"] == "sgd":
        return SGD(learning_rate=cfg["learning_rate"], decay=cfg["decay"] if cfg["decay"] else 0.0)
    if cfg["optimiser"] == "momentum":
        return SGDWithMomentum(
            learning_rate=cfg["learning_rate"],
            momentum=cfg["momentum"],
            decay=cfg["decay"] if cfg["decay"] else 0.0
        )
    if cfg["optimiser"] == "adam":
        return Adam(learning_rate=cfg["learning_rate"], decay=cfg["decay"] if cfg["decay"] else 0.0)
    raise ValueError("Unknown optimiser")

def build_layers(activation_name, l2, dropout_prob=None):
    """
    Builds network layers with optional inverted dropout.
    Dropout is applied only to hidden layers.
    Each layer gets its OWN activation and dropout instance.
    """
    if activation_name == "relu":
        act1 = ReLU()
        act2 = ReLU()
    elif activation_name == "sigmoid":
        act1 = Sigmoid()
        act2 = Sigmoid()
    else:
        raise ValueError("Unknown activation")

    dropout1 = Dropout(dropout_prob) if dropout_prob else None
    dropout2 = Dropout(dropout_prob) if dropout_prob else None

    return [
        {
            "n_inputs": 3072,
            "n_neurons": 1024,
            "activation": act1,
            "dropout": dropout1,
            "l2": l2
        },
        {
            "n_inputs": 1024,
            "n_neurons": 256,
            "activation": act2,
            "dropout": dropout2,
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

    layer_config = build_layers(cfg["activation"], l2, dropout_prob=cfg["dropout"])
    optimiser = build_optimiser(cfg)

    # ---- DataCatcher ----
    data_catcher = DataCatcher(
        base_dir=LOG_DIR,
        config={
            "experiment_name": cfg["name"],
            "metrics": True,
            "activation_logging": cfg["activation_logging"],
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

    seconds_per_epoch = duration / cfg["epochs"]
    epochs_per_second = cfg["epochs"] / duration

    data_catcher.save_final_results(
        config=cfg,
        final_test_accuracy=test_acc,
        total_time_seconds=duration
    )

    print(f"Finished {cfg['name']}")
    print(f"Final test accuracy   : {test_acc:.4f}")
    print(f"Total time taken     : {duration:.2f}s")
    print(f"Seconds per epoch    : {seconds_per_epoch:.3f}s")
    print(f"Epochs per second    : {epochs_per_second:.3f}")


print("\nAll experiments completed.")

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
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.1,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 256, "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 10, "activation": None, "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": True, "activation_type": "relu", "log_frequency": 10}
    },
    {
        "name": "T1b_sigmoid_sgd_bs256_epochs150",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.1,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "sigmoid", "dropout": None, "l2": None},
            {"n_neurons": 256, "activation": "sigmoid", "dropout": None, "l2": None},
            {"n_neurons": 10, "activation": None, "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": True, "activation_type": "sigmoid", "log_frequency": 10}
    },
    {
        "name": "T1d_relu_sgd_no_dropout",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 256, "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 10, "activation": None, "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": True, "activation_type": "relu", "log_frequency": 10}
    },
    {
        "name": "T1d_relu_sgd_dropout0.5",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": 0.5, "l2": None},
            {"n_neurons": 256, "activation": "relu", "dropout": 0.5, "l2": None},
            {"n_neurons": 10, "activation": None, "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": True, "activation_type": "relu", "log_frequency": 10}
    },
        {
        "name": "T1e_param_shallow_relu",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 256,  "activation": "relu", "dropout": None, "l2": None},
            {"n_neurons": 10,   "activation": None,   "dropout": None, "l2": None}
        ],
        "logging": {
            "metrics": True,
            "activations": False 
        }
    },
    {
        "name": "T1e_param_deep_relu",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "sgd", "decay": 0.0},
        "layers": [
            {'n_neurons': 1024, 'activation': 'relu', 'dropout': None, 'l2': None},
            {'n_neurons': 512, 'activation': 'relu', 'dropout': None, 'l2': None}, 
            {'n_neurons': 128, 'activation': 'relu', 'dropout': None, 'l2': None}, 
            {'n_neurons': 10, 'activation': None, 'dropout': None, 'l2': None}
        ],
        "logging": {
            "metrics": True,
            "activations": False 
        }
    },
    {
        "name": "T1G7_M1_relu_sgd_decay_l2",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "sgd", "decay": 1e-4},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": None, "l2": 0.0005},
            {"n_neurons": 256,  "activation": "relu", "dropout": None, "l2": 0.0005},
            {"n_neurons": 10,   "activation": None,   "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": False}
    },
    {
        "name": "T1G7_M2_relu_momentum_l2",
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 0.05,
        "optimiser": {"type": "momentum", "momentum": 0.9, "decay": 1e-4},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": None, "l2": 0.0005},
            {"n_neurons": 256,  "activation": "relu", "dropout": None, "l2": 0.0005},
            {"n_neurons": 10,   "activation": None,   "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": False}
    },
    {
        "name": "T1G7_M3_relu_adam_dropout_l2",
        "batch_size": 256,
        "epochs": 30,
        "learning_rate": 0.001,
        "optimiser": {"type": "adam", "beta1": 0.9, "beta2": 0.999, "decay": 0.0},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": 0.7, "l2": 0.0005},
            {"n_neurons": 256,  "activation": "relu", "dropout": 0.9, "l2": None},
            {"n_neurons": 10,   "activation": None,   "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": False}
    },
    {
        "name": "T1G7_FINAL_relu_momentum_dropout_decay",
        "batch_size": 256,
        "epochs": 60,
        "learning_rate": 0.08,
        "optimiser": {"type": "momentum", "momentum": 0.9, "decay": 5e-5},
        "layers": [
            {"n_neurons": 1024, "activation": "relu", "dropout": 0.85, "l2": 0.0005},
            {"n_neurons": 256,  "activation": "relu", "dropout": 0.95, "l2": None},
            {"n_neurons": 10,   "activation": None,   "dropout": None, "l2": None}
        ],
        "logging": {"metrics": True, "activations": False}
    },
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def get_activation(name):
    """Returns a NEW activation instance"""
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name is None:
        return None
    else:
        raise ValueError(f"Unknown activation: {name}")

def build_optimiser(cfg):
    """Build optimiser from config"""
    opt_cfg = cfg["optimiser"]
    lr = cfg["learning_rate"]
    
    if opt_cfg["type"] == "sgd":
        return SGD(learning_rate=lr, decay=opt_cfg.get("decay", 0.0))
    
    elif opt_cfg["type"] == "momentum":
        return SGDWithMomentum(
            learning_rate=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            decay=opt_cfg.get("decay", 0.0)
        )
    
    elif opt_cfg["type"] == "adam":
        return Adam(
            learning_rate=lr,
            beta1=opt_cfg.get("beta1", 0.9),
            beta2=opt_cfg.get("beta2", 0.999),
            decay=opt_cfg.get("decay", 0.0)
        )
    
    else:
        raise ValueError(f"Unknown optimiser: {opt_cfg['type']}")

def build_layers(cfg):
    """
    Build network layers from config.
    Each layer automatically gets its own activation and dropout instance.
    Input size is inferred from previous layer.
    """
    layers = []
    input_size = 3072  # CIFAR-10 flattened
    
    for layer_cfg in cfg["layers"]:
        # Create L2 regularizer if specified
        l2 = L2Regularizer(layer_cfg["l2"]) if layer_cfg["l2"] else None
        
        # Create dropout if specified
        dropout = Dropout(layer_cfg["dropout"]) if layer_cfg["dropout"] else None
        
        # Create activation (each layer gets its own instance)
        activation = get_activation(layer_cfg["activation"])
        
        layers.append({
            "n_inputs": input_size,
            "n_neurons": layer_cfg["n_neurons"],
            "activation": activation,
            "dropout": dropout,
            "l2": l2
        })
        
        input_size = layer_cfg["n_neurons"]
    
    return layers

def build_datacatcher(cfg):
    """Build DataCatcher from config"""
    log_cfg = cfg.get("logging", {})
    
    # Determine which layers to track (all layers with activations)
    layers_to_track = [
        i for i, layer in enumerate(cfg["layers"])
        if layer["activation"] is not None
    ]
    
    return DataCatcher(
        base_dir=LOG_DIR,
        config={
            "experiment_name": cfg["name"],
            "metrics": log_cfg.get("metrics", True),
            "activation_logging": log_cfg.get("activations", False),
            "activation_type": log_cfg.get("activation_type", "relu"),
            "activation_log_frequency": log_cfg.get("log_frequency", 10),
            "layers": layers_to_track
        }
    )

def print_experiment_summary(cfg):
    """Print a nice summary of the experiment configuration"""
    print(f"\n{'='*70}")
    print(f"Experiment: {cfg['name']}")
    print(f"{'='*70}")
    print(f"Training: {cfg['epochs']} epochs, batch size {cfg['batch_size']}")
    print(f"Optimiser: {cfg['optimiser']['type'].upper()}, LR={cfg['learning_rate']}")
    
    print(f"\nNetwork Architecture:")
    print(f"  Input: 3072 (CIFAR-10)")
    for i, layer in enumerate(cfg['layers']):
        act = layer['activation'] or 'None'
        drop = f"dropout={layer['dropout']}" if layer['dropout'] else ""
        l2 = f"l2={layer['l2']}" if layer['l2'] else ""
        extras = ", ".join(filter(None, [drop, l2]))
        extras = f" ({extras})" if extras else ""
        print(f"  Layer {i}: {layer['n_neurons']} neurons, {act}{extras}")
    print(f"{'='*70}\n")

# --------------------------------------------------
# Load data once
# --------------------------------------------------
print("Loading CIFAR-10...")
loader = Cifar10Loader(CIFAR_DIR)
X_train_raw, y_train_raw = loader.load_train_data()
X_test_raw, y_test_raw = loader.load_test_data()

preprocessor = Preprocessor()
X_train, y_train, X_val, y_val = preprocessor.preprocess_train_data(
    X_train_raw, y_train_raw, train_ratio=0.9
)
X_test, y_test = preprocessor.preprocess_test_data(X_test_raw, y_test_raw)

print(f"Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# --------------------------------------------------
# Run experiments
# --------------------------------------------------
results = []

for cfg in EXPERIMENTS:
    print_experiment_summary(cfg)
    
    # Reset seed for reproducibility
    np.random.seed(42)
    
    # Build components
    layer_config = build_layers(cfg)
    optimiser = build_optimiser(cfg)
    data_catcher = build_datacatcher(cfg)
    
    # Create network
    network = NeuralNetwork(layer_config=layer_config, optimiser=optimiser)
    
    # Train
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
    
    # Evaluate
    test_acc = network.accuracy(X_test, y_test)
    
    # Save logs
    data_catcher.save_final_results(
        config=cfg,
        final_test_accuracy=test_acc,
        total_time_seconds=duration
    )
    
    # Store results
    result = {
        "name": cfg["name"],
        "test_accuracy": test_acc,
        "duration": duration,
        "epochs": cfg["epochs"]
    }
    results.append(result)
    
    # Print summary
    print(f"\n✓ Completed: {cfg['name']}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Total Time: {duration:.2f}s ({duration/cfg['epochs']:.3f}s/epoch)")
    print(f"  Speed: {cfg['epochs']/duration:.3f} epochs/sec\n")

# --------------------------------------------------
# Final summary
# --------------------------------------------------
print(f"\n{'='*70}")
print("ALL EXPERIMENTS COMPLETED")
print(f"{'='*70}")
print(f"{'Experiment':<40} {'Test Acc':<12} {'Time (s)':<12}")
print(f"{'-'*70}")
for r in results:
    print(f"{r['name']:<40} {r['test_accuracy']:<12.4f} {r['duration']:<12.2f}")
print(f"{'='*70}\n")
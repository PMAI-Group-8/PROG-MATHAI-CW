import numpy as np
import time
np.random.seed(42)

from Cifar10Loader import Cifar10Loader
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
from layers.activation_functions import ReLU, Sigmoid
from layers.dropout import Dropout
from layers.optimisers import SGD, SGDWithMomentum, Adam
from layers.l2_regulariser import L2Regularizer

# ----------------------------
# Config
# ----------------------------
CIFAR_DIR = "./dataset/cifar10"
EPOCHS = 101
BATCH_SIZE = 1024

# ----------------------------
# Load data
# ----------------------------
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


# =========================================================
# Experiment runner
# =========================================================
def run_experiment(name, task_code, activation_cls, optimiser, dropout=None, l2=None):
    print("\n" + "=" * 60)
    print(f"RUN : {name}")
    print(f"TASK: {task_code}")
    print("=" * 60)

    layer_config = [
        {
            'n_inputs': 3072,
            'n_neurons': 512,
            'activation': activation_cls(),
            'dropout': Dropout(dropout.keep_prob) if dropout else None,
            'l2': l2
        },
        {
            'n_inputs': 512,
            'n_neurons': 128,
            'activation': activation_cls(),
            'dropout': Dropout(dropout.keep_prob) if dropout else None,
            'l2': l2
        },
        {
            'n_inputs': 128,
            'n_neurons': 10,
            'activation': None,
            'dropout': None,
            'l2': None
        }
    ]

    nn = NeuralNetwork(layer_config, optimiser)

    # -------- Timing start --------
    start_time = time.perf_counter()

    nn.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # -------- Timing end --------
    elapsed = time.perf_counter() - start_time
    seconds_per_epoch = elapsed / EPOCHS

    acc = nn.accuracy(X_test, y_test)

    n_samples = X_train.shape[0]
    iters_per_epoch = int(np.ceil(n_samples / BATCH_SIZE))
    total_iterations = iters_per_epoch * EPOCHS
    iters_per_sec = total_iterations / max(elapsed, 1e-8)


    print(f"[{task_code}] Test accuracy : {acc * 100:.2f}%")
    print(f"[{task_code}] Time taken    : {elapsed:.2f} seconds")
    print(f"[{task_code}] Speed         : {seconds_per_epoch:.2f} seconds/epoch")
    print(f"[{task_code}] Iterations/sec  : {iters_per_sec:.2f}")


    return acc, elapsed, seconds_per_epoch


# =========================================================
# Task 1b – Activation comparison
# =========================================================
sgd = SGD(learning_rate=0.1, decay=2e-3)

relu_acc, relu_time, relu_speed = run_experiment(
    "ReLU activation",
    "T1-b",
    ReLU,
    sgd
)

sigmoid_acc, sigmoid_time, sigmoid_speed = run_experiment(
    "Sigmoid activation",
    "T1-b",
    Sigmoid,
    sgd
)

# =========================================================
# Task 1f – Optimiser comparison
# =========================================================
momentum = SGDWithMomentum(learning_rate=0.05, momentum=0.9)
adam = Adam(learning_rate=0.001)

momentum_acc, momentum_time, momentum_speed = run_experiment(
    "SGD with Momentum",
    "T1-f",
    ReLU,
    momentum
)

adam_acc, adam_time, adam_speed = run_experiment(
    "Adam Optimiser",
    "T1-f",
    ReLU,
    adam
)

# =========================================================
# Task 1d – Dropout
# =========================================================
dropout_acc, dropout_time, dropout_speed = run_experiment(
    "ReLU + Dropout",
    "T1-d",
    ReLU,
    sgd,
    dropout=Dropout(keep_prob=0.8)
)

# =========================================================
# Task 1e – L2 Regularisation
# =========================================================
l2_acc, l2_time, l2_speed = run_experiment(
    "ReLU + L2",
    "T1-e",
    ReLU,
    sgd,
    l2=L2Regularizer(lam=1e-4)
)

# =========================================================
# Task 1f – Learning rate decay comparison
# =========================================================
print("\n================ DECAY EXPERIMENTS ================\n")

# ---- SGD decay vs no decay ----
sgd_no_decay = SGD(learning_rate=0.1, decay=0.0)
sgd_decay = SGD(learning_rate=0.1, decay=2e-3)

sgd_no_decay_acc, _, _ = run_experiment(
    "SGD (no decay)",
    "T1-f",
    ReLU,
    sgd_no_decay
)

sgd_decay_acc, _, _ = run_experiment(
    "SGD (with decay)",
    "T1-f",
    ReLU,
    sgd_decay
)

# ---- Momentum decay vs no decay ----
momentum_no_decay = SGDWithMomentum(learning_rate=0.05, momentum=0.9, decay=0.0)
momentum_decay = SGDWithMomentum(learning_rate=0.05, momentum=0.9, decay=2e-3)

momentum_no_decay_acc, _, _ = run_experiment(
    "Momentum (no decay)",
    "T1-f",
    ReLU,
    momentum_no_decay
)

momentum_decay_acc, _, _ = run_experiment(
    "Momentum (with decay)",
    "T1-f",
    ReLU,
    momentum_decay
)

# ---- Adam decay vs no decay ----
adam_no_decay = Adam(learning_rate=0.001, decay=0.0)
adam_decay = Adam(learning_rate=0.001, decay=1e-4)

adam_no_decay_acc, _, _ = run_experiment(
    "Adam (no decay)",
    "T1-f",
    ReLU,
    adam_no_decay
)

adam_decay_acc, _, _ = run_experiment(
    "Adam (with decay)",
    "T1-f",
    ReLU,
    adam_decay
)

# ----------------------------
# Summary (copy into report)
# ----------------------------
print("\n=== TASK 1 SUMMARY ===")
print(f"ReLU            : {relu_acc * 100:.2f}%")
print(f"Sigmoid         : {sigmoid_acc * 100:.2f}%")
print(f"Momentum        : {momentum_acc * 100:.2f}%")
print(f"Adam            : {adam_acc * 100:.2f}%")
print(f"Dropout         : {dropout_acc * 100:.2f}%")
print(f"L2 Regulariser  : {l2_acc * 100:.2f}%")
print(f"SGD no decay    : {sgd_no_decay_acc * 100:.2f}%")
print(f"SGD with decay  : {sgd_decay_acc * 100:.2f}%")
print(f"Momentum no decay : {momentum_no_decay_acc * 100:.2f}%")
print(f"Momentum with decay : {momentum_decay_acc * 100:.2f}%")
print(f"Adam no decay   : {adam_no_decay_acc * 100:.2f}%")
print(f"Adam with decay : {adam_decay_acc * 100:.2f}%")
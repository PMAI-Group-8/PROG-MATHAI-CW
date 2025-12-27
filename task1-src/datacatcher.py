import os
import csv
import numpy as np


class DataCatcher:
    """
    DataCatcher
    ===========
    Experiment logger for NumPy-based neural networks.

    Features
    --------
    - Creates a new folder per test run
    - Logs training metrics (loss, acc, val acc)
    - Logs per-layer dead-neuron statistics
    - CSV output is Excel heatmap friendly
    """

    def __init__(self, base_dir="logs", config=None):
        if config is None:
            raise ValueError("DataCatcher requires a config dictionary")

        # ---- Resolve base directory relative to project root ----
        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        self.base_dir = os.path.join(PROJECT_ROOT, base_dir)

        # ---- Experiment directory ----
        self.exp_name = config.get("experiment_name", "experiment")
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)


        # ---- Metrics CSV ----
        self.metrics_enabled = config.get("metrics", True)
        self.metrics_path = os.path.join(self.exp_dir, "metrics.csv")

        if self.metrics_enabled:
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "loss",
                    "train_accuracy",
                    "validation_accuracy"
                ])

        # ---- Activation logging ----
        self.activation_enabled = config.get("activation_logging", False)
        self.activation_type = config.get("activation_type", "relu")
        self.layers_to_track = config.get("layers", [])

        self.activation_buffers = {}

        if self.activation_enabled:
            for lid in self.layers_to_track:
                path = os.path.join(
                    self.exp_dir,
                    f"layer_{lid}_dead_neurons.csv"
                )
                self.activation_buffers[lid] = {
                    "path": path,
                    "rows": []
                }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def log_metrics(self, epoch, loss, train_acc, val_acc):
        if not self.metrics_enabled:
            return

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                float(loss),
                float(train_acc),
                float(val_acc)
            ])

    # ------------------------------------------------------------------
    # Activation logging
    # ------------------------------------------------------------------
    def log_activations(self, epoch, layers):
        if not self.activation_enabled:
            return

        for lid in self.layers_to_track:
            layer = layers[lid]

            if not hasattr(layer, "A"):
                continue  # safety guard

            A = layer.A

            if self.activation_type == "relu":
                dead = np.mean(A <= 0, axis=0)

            elif self.activation_type == "sigmoid":
                dead = np.mean((A < 0.05) | (A > 0.95), axis=0)

            else:
                raise ValueError("Unsupported activation type")

            self.activation_buffers[lid]["rows"].append(dead)

    # ------------------------------------------------------------------
    # Flush activation CSVs
    # ------------------------------------------------------------------
    def save_activation_logs(self):
        """
        Writes activation buffers to CSV files.
        Rows = neurons
        Columns = epochs
        """
        if not self.activation_enabled:
            return

        for lid, buffer in self.activation_buffers.items():
            data = np.array(buffer["rows"]).T  # neurons x epochs

            with open(buffer["path"], "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                header = ["neuron"] + [f"epoch_{i}" for i in range(data.shape[1])]
                writer.writerow(header)

                # Rows
                for i, row in enumerate(data):
                    writer.writerow([i] + list(row))
    # ------------------------------------------------------------------
    # End-of-test summary
    # ------------------------------------------------------------------
    def save_final_results(
        self,
        config,
        final_test_accuracy,
        total_time_seconds
    ):
        """
        Saves experiment configuration and final results to CSV.
        """

        path = os.path.join(self.exp_dir, "end_of_test_results.csv")

        epochs = config.get("epochs", 1)
        seconds_per_epoch = total_time_seconds / epochs
        epochs_per_second = epochs / total_time_seconds

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])

            # ---- Experiment config ----
            for k, v in config.items():
                writer.writerow([k, v])

            # ---- Final results ----
            writer.writerow(["final_test_accuracy", float(final_test_accuracy)])
            writer.writerow(["total_time_seconds", float(total_time_seconds)])
            writer.writerow(["seconds_per_epoch", float(seconds_per_epoch)])
            writer.writerow(["epochs_per_second", float(epochs_per_second)])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def summary(self):
        print(f"[DataCatcher] Experiment: {self.exp_name}")
        print(f"[DataCatcher] Directory : {self.exp_dir}")
        print(f"[DataCatcher] Metrics   : {self.metrics_enabled}")
        print(f"[DataCatcher] Activations: {self.activation_enabled}")

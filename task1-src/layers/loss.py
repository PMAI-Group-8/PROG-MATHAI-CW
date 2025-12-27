import numpy as np

np.random.seed(42)

class SoftmaxCrossEntropy:
    """
    Combined Softmax activation and Cross-Entropy loss.
    Implemented together for numerical stability and a simpler backward pass.
    """

    def _softmax(self, logits):
        # Shift logits by max value per sample for numerical stability
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        # Compute class probabilities using softmax
        self.probs = self._softmax(logits)

        # Store true labels for backward pass
        self.y_true = y_true
        self.batch_size = y_true.shape[0]

        # Cross-entropy loss (small constant added to avoid log(0))
        return -np.mean(
            np.sum(y_true * np.log(self.probs + 1e-9), axis=1)
        )

    def backward(self):
        # Gradient of softmax + cross-entropy with respect to logits
        return (self.probs - self.y_true) / self.batch_size
import numpy as np

np.random.seed(42)

class SoftmaxCrossEntropy:
    """
    Combined Softmax activation and Cross-Entropy loss.
    Implemented together for numerical stability and a simpler backward pass.
    """

    def _softmax(self, logits):
        # Shift logits by max value per sample for numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        # Compute class probabilities using softmax
        self.probs = self._softmax(logits)

        # Store for backward pass
        self.y_true = y_true
        self.batch_size = y_true.shape[0]

        # Cross-entropy loss using np.clip for numerical stability
        return -np.sum(y_true * np.log(np.clip(self.probs, 1e-15, 1.0))) / self.batch_size

    def backward(self):
        # Gradient of softmax + cross-entropy with respect to logits
        return (self.probs - self.y_true) / self.batch_size
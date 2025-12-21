import numpy as np

np.random.seed(42)

class SoftmaxCrossEntropy:
    def _softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        self.probs = self._softmax(logits)
        self.y_true = y_true
        self.batch_size = y_true.shape[0]

        return -np.mean(
            np.sum(y_true * np.log(self.probs + 1e-9), axis=1)
        )

    def backward(self):
        return (self.probs - self.y_true) / self.batch_size
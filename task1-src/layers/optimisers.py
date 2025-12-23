import numpy as np
 
np.random.seed(42)
 
class Optimiser:
    def update(self, layer, dW, db):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
 
class SGD(Optimiser):
    def __init__(self, learning_rate=0.01, decay=0.0):
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
 
    def update(self, layer, dW, db):
        if self.decay > 0:
            self.learning_rate = self.initial_lr / (1 + self.decay * self.iterations)
 
        layer.W -= self.learning_rate * dW
        layer.b -= self.learning_rate * db
 
        self.iterations += 1
 
class SGDWithMomentum(Optimiser):
    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0.0):
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0
 
        self.vW = {}
        self.vb = {}
 
    def update(self, layer, dW, db):
        if self.decay > 0:
            self.learning_rate = self.initial_lr / (1 + self.decay * self.iterations)
 
        lid = layer.layer_id
 
        if lid not in self.vW:
            self.vW[lid] = np.zeros_like(dW)
            self.vb[lid] = np.zeros_like(db)
 
        self.vW[lid] = self.momentum * self.vW[lid] - self.learning_rate * dW
        self.vb[lid] = self.momentum * self.vb[lid] - self.learning_rate * db
 
        layer.W += self.vW[lid]
        layer.b += self.vb[lid]
 
        self.iterations += 1
 
class Adam(Optimiser):
    def __init__(
        self, learning_rate=0.001,
        beta1=0.9, beta2=0.999,
        epsilon=1e-8, decay=0.0
    ):
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0
 
        self.mW = {}
        self.mb = {}
        self.vW = {}
        self.vb = {}
 
    def update(self, layer, dW, db):
        if self.decay > 0:
            self.learning_rate = self.initial_lr / (1 + self.decay * self.iterations)
 
        lid = layer.layer_id
 
        if lid not in self.mW:
            self.mW[lid] = np.zeros_like(dW)
            self.mb[lid] = np.zeros_like(db)
            self.vW[lid] = np.zeros_like(dW)
            self.vb[lid] = np.zeros_like(db)
 
        # First moment (mean)
        self.mW[lid] = self.beta1 * self.mW[lid] + (1 - self.beta1) * dW
        self.mb[lid] = self.beta1 * self.mb[lid] + (1 - self.beta1) * db
 
        # Second moment (variance)
        self.vW[lid] = self.beta2 * self.vW[lid] + (1 - self.beta2) * (dW ** 2)
        self.vb[lid] = self.beta2 * self.vb[lid] + (1 - self.beta2) * (db ** 2)
 
        # Bias correction
        t = self.iterations + 1
        mW_hat = self.mW[lid] / (1 - self.beta1 ** t)
        mb_hat = self.mb[lid] / (1 - self.beta1 ** t)
        vW_hat = self.vW[lid] / (1 - self.beta2 ** t)
        vb_hat = self.vb[lid] / (1 - self.beta2 ** t)
 
        # Update params
        layer.W -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
 
        self.iterations += 1
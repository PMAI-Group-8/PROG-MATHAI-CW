import os
import pickle
import numpy as np

class Cifar10Loader:
    def __init__(self, cifar_dir):
        self.cifar_dir = cifar_dir

    def _load_batch(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        X = batch[b'data']          # (10000, 3072), uint8
        y = np.array(batch[b'labels'])
        return X, y

    def load_train_data(self):
        X_list, y_list = [], []

        for i in range(1, 6):
            file = os.path.join(self.cifar_dir, f'data_batch_{i}')
            X, y = self._load_batch(file)
            X_list.append(X)
            y_list.append(y)

        X_train = np.vstack(X_list)        # (50000, 3072)
        y_train = np.concatenate(y_list)   # (50000,)

        return X_train, y_train

    def load_test_data(self):
        file = os.path.join(self.cifar_dir, 'test_batch')
        return self._load_batch(file)
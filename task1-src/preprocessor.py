import numpy as np

np.random.seed(42)


class Preprocessor:
    def __init__(self):
        pass

    def normalize(self, X_data):
        """Scale input data to [0, 1]."""
        return X_data.astype(np.float32) / 255.0
    
    def shuffle(self, X_data, y_data):
        ''' Shuffle the input data X_data and labels y_data in unison.'''
        assert len(X_data) == len(y_data), "Data and labels must have the same length."
        p = np.random.permutation(len(X_data))
        return X_data[p], y_data[p]
    
    def split(self, X_data, y_data, train_ratio = 0.8):
        ''' Split the input data X_data and labels y_data into training and validation sets.'''
        assert len(X_data) == len(y_data), "Data and labels must have the same length."
        split_index = int(len(X_data) * train_ratio)
        X_train, X_val = X_data[:split_index], X_data[split_index:]
        y_train, y_val = y_data[:split_index], y_data[split_index:]
        return X_train, y_train, X_val, y_val
    
    def preprocess_train_data(self, X_data, y_data, train_ratio = 0.8):
        ''' Complete preprocessing pipeline: normalize, shuffle, and split the data.'''
        X_normalized = self.normalize(X_data)
        X_shuffled, y_shuffled = self.shuffle(X_normalized, y_data)
        return self.split(X_shuffled, y_shuffled, train_ratio)
    
    def preprocess_test_data(self, X_test, y_test):
        X_normalized = self.normalize(X_test)
        X_shuffled, y_shuffled = self.shuffle(X_normalized, y_test)
        return X_shuffled, y_shuffled
    
class PreprocessorAdvanced():
    def __init__(self):
        self.mean = None
        self.std = None

    def normalize(self, X):
        return X.astype(np.float32) / 255.0

    def standardize_fit(self, X):
        """Fit per-pixel mean/std on training data."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return (X - self.mean) / self.std

    def standardize_transform(self, X):
        """Apply stored mean/std to val/test data."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Must fit standardisation on training data first")
        return (X - self.mean) / self.std

    def shuffle(self, X, y):
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        return X[p], y[p]

    def split(self, X, y, train_ratio=0.8):
        split_idx = int(len(X) * train_ratio)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def preprocess_train_data(self, X, y, train_ratio=0.8):
        X = self.normalize(X)
        X = X.reshape(X.shape[0], -1)

        X, y = self.shuffle(X, y)
        X_train, y_train, X_val, y_val = self.split(X, y, train_ratio)

        X_train = self.standardize_fit(X_train)
        X_val   = self.standardize_transform(X_val)

        return X_train, y_train, X_val, y_val

    def preprocess_test_data(self, X_test, y_test):
        X = self.normalize(X_test)
        X = X.reshape(X.shape[0], -1)
        X = self.standardize_transform(X)

        return X, y_test
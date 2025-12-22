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
    
    def split(self, X_data, y_data, validation_ratio = 0.8):
        ''' Split the input data X_data and labels y_data into training and validation sets.'''
        assert len(X_data) == len(y_data), "Data and labels must have the same length."
        split_index = int(len(X_data) * validation_ratio)
        X_train, X_val = X_data[:split_index], X_data[split_index:]
        y_train, y_val = y_data[:split_index], y_data[split_index:]
        return X_train, y_train, X_val, y_val
    
    def preprocess_train_data(self, X_data, y_data, validation_ratio = 0.8):
        ''' Complete preprocessing pipeline: normalize, shuffle, and split the data.'''
        X_normalized = self.normalize(X_data)
        X_shuffled, y_shuffled = self.shuffle(X_normalized, y_data)
        return self.split(X_shuffled, y_shuffled, validation_ratio)
    
    def preprocess_test_data(self, X_test, y_test):
        X_normalized = self.normalize(X_test)
        X_shuffled, y_shuffled = self.shuffle(X_normalized, y_test)
        return X_shuffled, y_shuffled
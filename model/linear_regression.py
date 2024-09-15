"""ML module that implements the Linear Regression Model from scratch using Numpy"""

import numpy as np


class LinearRegression:
    """Implementation of Linear Regression Model"""

    def __init__(self, lr: float = 0.01, n_iters: int = 1000) -> None:
        self.lr: float = lr
        self.n_iters: int = n_iters
        self.weights: np.ndarray
        self.bias: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the training data to the model"""
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("X and y should be numpy arrays")

        n_samples, n_features = X.shape
        # Initialize the weights randomly between -1 and 1
        self.weights: np.ndarray = np.random.uniform(-1.0, 1.0, n_features)
        # Initialize bias to a value of 0
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Get the predictions
            y_predicted = self.predict(X)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict the output for the given data"""
        return np.dot(X, self.weights) + self.bias

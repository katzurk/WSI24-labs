import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.weights = None
        self.bias = None
        self.iterations = iterations
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize(self, X):
        self.weights = np.zeros(X.shape[1]) # weight for each input parameter
        self.bias = 0

    def fit(self, X, y):
        self.initialize(X)

        for i in range(self.iterations):
            linear = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(linear)

            diff = pred - y

            dw = np.dot(X.T, diff) / np.shape(X)[0]
            db = np.sum(diff) / np.shape(X)[0]

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y = np.where(pred > 0.5, 1, 0)
        return y
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.weights = None
        self.bias = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.costs =[]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize(self, X):
        self.weights = np.zeros(X.shape[1]) # weight for each input parameter
        self.bias = 0

    def cost(self, X, y, pred):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)

        cost = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / np.shape(X)[0]
        return cost

    def fit(self, X, y):
        self.initialize(X)

        for i in range(self.iterations):
            linear = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(linear)
            cost = self.cost(X, y, pred)
            self.costs.append(cost)

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

    def accuracy(self, y, y_test):
        score = accuracy_score(y_test, y)
        return score

    def F1(self, y, y_test):
        score = f1_score(y_test, y)
        return score

    def auroc(self, y, y_test):
        score = roc_auc_score(y_test, y)
        return score

    def score_comparison(self, y, y_test):
        scores = {
            "accuracy": [self.accuracy(y, y_test)],
            "F1": [self.F1(y, y_test)],
            "auroc": [self.auroc(y, y_test)]
        }
        df =pd.DataFrame.from_dict(scores)
        return df

    def weights_graph(self):
        plt.bar(range(len(self.weights)), self.weights)
        plt.show()

    def costs_graph(self):
        plt.plot(self.costs)
        plt.show()

    def roc_curve_graph(self, X, y_test):
        linear = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        print(fpr)
        print(tpr)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
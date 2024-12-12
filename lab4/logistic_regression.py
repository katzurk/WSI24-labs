import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.theta = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.costs =[]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize(self, X):
        self.theta = np.zeros(X.shape[1]) # bias and weight for each input parameter

    def cost(self, X, y, pred):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)

        cost = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / np.shape(X)[0]
        return cost

    def gradient_descent(self, X, y):
        for i in range(self.iterations):
            linear = np.dot(X, self.theta)
            pred = self.sigmoid(linear)
            cost = self.cost(X, y, pred)
            self.costs.append(cost)
            gradient = (np.dot(X.transpose(), (pred - y))) / np.shape(X)[0]
            self.theta -= self.learning_rate * gradient


    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.initialize(X)

        self.gradient_descent(X, y)

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        linear = np.dot(X, self.theta)
        pred = self.sigmoid(linear)
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
        plt.bar(range(len(self.theta)), self.theta)
        plt.title("Weight graph")
        plt.savefig("weights.jpg")
        plt.close()

    def costs_graph(self):
        plt.plot(self.costs)
        plt.title("Costs across iterations")
        plt.ylim(bottom=0.001)
        plt.savefig("costs.jpg")
        plt.close()

    def roc_curve_graph(self, X, y_test):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        linear = np.dot(X, self.theta)
        pred = self.sigmoid(linear)
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.plot(fpr, tpr)
        plt.title("Roc Curve Graph")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig("roc_curve.jpg")
        plt.close()
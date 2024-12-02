from data import split_data
from logistic_regression import LogisticRegression
import numpy as np

if __name__ == "__main__":
    np.random.seed(0)
    X, X_test, y, y_test = split_data()
    lg = LogisticRegression(0.01, 10000)
    lg.fit(X, y)
    y = lg.predict(X_test)
    print(lg.accuracy(y, y_test))
    print(lg.AUROC(y, y_test))
    print(lg.F1(y, y_test))
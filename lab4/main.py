from data import split_data
from logistic_regression import LogisticRegression

if __name__ == "__main__":
    X, X_test, y, y_test = split_data()
    lg = LogisticRegression(0.00001, 500)
    lg.fit(X, y)
    print(lg.predict(X_test))
    print(y_test)

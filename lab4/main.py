from data import import_data
from logistic_regression import LogisticRegression
import numpy as np
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    np.random.seed(0)
    X, X_test, y, y_test = import_data(exclude_inputs=True, n_columns=3)
    lg = LogisticRegression(0.06, 10000)
    lg.fit(X, y)
    y = lg.predict(X_test)
    print(lg.score_comparison(y, y_test))
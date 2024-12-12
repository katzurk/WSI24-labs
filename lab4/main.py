from data import import_data
from logistic_regression import LogisticRegression
import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--exclude-columns", type=int, default=0)
    parser.add_argument("--correlations", type=int, choices=[0, 1, 2], default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    X, X_test, y, y_test = import_data(args.exclude_columns, args.normalize, args.seed, args.correlations)
    lg = LogisticRegression(args.learning_rate, args.iterations)
    lg.fit(X, y)
    y = lg.predict(X_test)
    print(lg.score_comparison(y, y_test))
    lg.roc_curve_graph(X_test, y_test)
    lg.weights_graph()
    lg.costs_graph()
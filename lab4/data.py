from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import *

def get_data():
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    y.Diagnosis = [1 if value == "M" else 0 for value in y.Diagnosis]
    return X, y

def normalize(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split_data():
    X, y = get_data();
    X = normalize(X)
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data()
print(X_test)
print(y_test)
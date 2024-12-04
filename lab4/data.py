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

    y.loc[:, 'Diagnosis'] = [1 if value == "M" else 0 for value in y['Diagnosis']]
    y['Diagnosis'] = y['Diagnosis'].astype(int)
    y = y.values.ravel()

    return X, y

def exclude_columns(X, y, n_columns):
    excluded = np.random.choice(X.columns, n_columns, replace=False)
    X = X.drop(columns = excluded)
    return X, y

def normalize(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=1)

def import_data(exclude_inputs = False, n_columns = 0, to_normalize = True):
    X, y = get_data()
    if exclude_inputs: X, y = exclude_columns(X, y, n_columns)
    if to_normalize: X, y = normalize(X, y)
    X, X_test, y, y_test = split_data(X, y)
    return X, X_test, y, y_test

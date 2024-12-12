from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
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
    print(X.columns)
    print()
    return X, y

def correlations(X, y):
    return X.corrwith(pd.Series(y))

def delete_columns_low_corr(X, y):
    corr = correlations(X, y)
    delete_columns = corr[corr.abs() < 0.1].index.tolist()
    X = X.drop(columns=delete_columns)
    return X, y

def delete_columns_high_corr(X, y):
    corr = correlations(X, y)
    delete_columns = corr[corr.abs() > 0.5].index.tolist()
    X = X.drop(columns=delete_columns)
    return X, y

def normalize(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_data(X, y, seed):
    state = np.random.get_state()
    if seed is not None: np.random.seed(seed)
    return train_test_split(X, y, test_size=0.25, random_state=1)
    np.random.set_state(state)

def import_data(n_columns = 0, to_normalize = 1, seed=None, corr=0):
    X, y = get_data()
    if corr == 1: X, y = delete_columns_low_corr(X, y)
    if corr == 2: X, y = delete_columns_high_corr(X, y)
    if n_columns != 0: X, y = exclude_columns(X, y, n_columns)
    if to_normalize: X, y = normalize(X, y)
    X, X_test, y, y_test = split_data(X, y, seed)
    return X, X_test, y, y_test

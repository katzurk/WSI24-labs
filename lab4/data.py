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

    return X, y

def normalize(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y.loc[:, 'Diagnosis'] = [1 if value == "M" else 0 for value in y['Diagnosis']]
    y['Diagnosis'] = y['Diagnosis'].astype(int)

    return X, y

def split_data():
    X, y = get_data();
    X, y = normalize(X, y)
    y = y.values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    return X_train, X_test, y_train, y_test

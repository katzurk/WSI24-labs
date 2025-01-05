import pandas as pd
from pgmpy.estimators import PC, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
import seaborn as sns
from matplotlib import pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("pgmpy").setLevel(logging.WARNING)

def get_data():
    df = pd.read_csv("US_Crime_DataSet.csv")
    df = df[["Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"]]
    df = df[~df.isin(['Unknown']).any(axis=1)]
    df = df.head(100)

    for col in ["Victim Sex", "Victim Race", "Perpetrator Sex", "Perpetrator Race", "Relationship", "Weapon"]:
        df[col] = df[col].astype("category")

    df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    df["Perpetrator Age"] = pd.to_numeric(df["Perpetrator Age"], errors="coerce")

    # Drop rows with any missing values after conversion
    df.dropna(inplace=True)

    return df

data = get_data()
print(data)


est = PC(data)
skel, seperating_sets = est.build_skeleton(significance_level=0.01)
pdag = est.skeleton_to_pdag(skel, seperating_sets)
dag = pdag.to_dag()

model = BayesianNetwork(dag.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

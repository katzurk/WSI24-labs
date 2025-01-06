import pandas as pd
from pgmpy.estimators import PC, MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter
import seaborn as sns
import multiprocessing
from matplotlib import pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("pgmpy").setLevel(logging.WARNING)

def get_data():
    df = pd.read_csv("US_Crime_DataSet.csv")
    df = df[["Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"]]
    df = df[~df.isin(['Unknown']).any(axis=1)]

    for col in ["Victim Sex", "Victim Race", "Perpetrator Sex", "Perpetrator Race", "Relationship", "Weapon"]:
        df[col] = df[col].astype("category")

    df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    df["Perpetrator Age"] = pd.to_numeric(df["Perpetrator Age"], errors="coerce")

    df.dropna(inplace=True)

    return df

data = get_data()

# est = PC(data)
# skel, separating_sets = est.build_skeleton(significance_level=0.05, max_cond_vars=2)
# pdag = est.skeleton_to_pdag(skel, separating_sets)
# dag = pdag.to_dag()

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))

print(best_model.edges())

model = BayesianNetwork(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

for cpd in model.get_cpds():
    print(cpd)

# writer = XMLBIFWriter(model)
# writer.write_xmlbif('BN_Model_US_Crime.xml')
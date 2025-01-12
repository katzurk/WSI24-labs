import pandas as pd
from pgmpy.estimators import PC, MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.readwrite import XMLBIFWriter
from pgmpy.inference import VariableElimination
from matplotlib import pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("pgmpy").setLevel(logging.WARNING)

def get_data():
    df = pd.read_csv("US_Crime_DataSet.csv")
    # df = df[(df["Year"] >= 1980) & (df["Year"] <= 2000)]
    df.dropna()
    df = df[~df.applymap(lambda x: x == 'Unknown').any(axis=1)]

    df = df[["Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"]]

    # filtered_df = df[(df['Victim Age'] == 998) & (df['Perpetrator Age'] == 0)]
    # print(filtered_df)

    for col in ["Victim Sex", "Victim Race", "Perpetrator Sex", "Perpetrator Race", "Relationship", "Weapon"]:
        df[col] = df[col].astype("category")

    df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    df["Perpetrator Age"] = pd.to_numeric(df["Perpetrator Age"], errors="coerce")

    return df

data = get_data()

# est = PC(data)
# dag = est.estimate(significance_level=0.01)
# skel, seperating_sets = est.build_skeleton(significance_level=0.01)
# pdag = est.skeleton_to_pdag(skel, seperating_sets)
# dag = pdag.to_dag()

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))

# print(dag.edges())

model = BayesianNetwork(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

# for cpd in model.get_cpds():
#     print(cpd)

# model_daft = model.to_daft()
# # To open the plot
# model_daft.render()
# # Save the plot
# model_daft.savefig('sachs.png')
# writer = XMLBIFWriter(model)
# writer.write_xmlbif('BN_Model_US_Crime.xml')

infer = VariableElimination(model)
q = infer.map_query(variables=['Victim Age', 'Victim Sex', 'Perpetrator Age', 'Relationship'], evidence={'Perpetrator Sex': 'Female'})
print(q)

graph = model.to_graphviz()
graph.draw('model.png', prog='dot')

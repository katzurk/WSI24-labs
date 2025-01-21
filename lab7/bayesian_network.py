import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore, TreeSearch
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
    df.dropna()
    df = df[~df.applymap(lambda x: x == 'Unknown').any(axis=1)]

    df = df[["Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"]]
    for col in ["Victim Sex", "Victim Race", "Perpetrator Sex", "Perpetrator Race", "Relationship", "Weapon"]:
        df[col] = df[col].astype("category")

    df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    df["Perpetrator Age"] = pd.to_numeric(df["Perpetrator Age"], errors="coerce")

    df = df[(df["Victim Age"] != 998) & (df["Perpetrator Age"] != 0)]

    return df

def create_bayesian_network(data):
    hc = HillClimbSearch(data)
    graph = hc.estimate(scoring_method=BicScore(data))
    model = BayesianNetwork(graph.edges())

    model.fit(data, estimator=MaximumLikelihoodEstimator)

    for i, cpd in enumerate(model.get_cpds()):
        cpd.to_csv(filename="distribution/" + str(i) + "_condition.csv")
        dist = cpd.marginalize(cpd.get_evidence(), inplace=False)
        dist.to_csv(filename="distribution/" + str(i) + ".csv")
        visualize_distribution(dist)

    writer = XMLBIFWriter(model)
    writer.write_xmlbif('BN_Model_US_Crime.xml')

    return model

def visualize_network(model):
    graph = model.to_graphviz()
    graph.draw('model.png', prog='dot')

def visualize_distribution(cpd):
    var = cpd.variables[0]
    state = cpd.state_names[var]

    plt.figure(figsize=(10, 6))
    plt.barh(state, cpd.values)
    plt.title(f"Distribution of {var}")
    plt.xlabel(str(var))
    plt.ylabel("Distribution")

    plt.savefig(f"bars/{var}.jpg")

def create_inference(model, statement):
    if len(statement) < 8:
        raise ValueError

    variables = ["Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"]
    evidence = {}

    for i, var in enumerate(statement):
        if var != "?":
            evidence[variables[i]] = var if not var.isdigit() else int(var)

    variables = [var for var in variables if var not in evidence.keys()]

    infer = VariableElimination(model)
    q = infer.map_query(variables=variables, evidence=evidence)
    q.update(evidence)
    return q
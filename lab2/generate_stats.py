import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from evolutionary_algorithm import Evolutionary_Algorithm
from solution_utils import *


def generate_plot(ea):
    best, grade = zip(*(ea.development))
    plt.plot(np.arange(len(best)), grade, marker="o")
    plt.title("Best solution for each generation")
    plt.xlabel("Generation number")
    plt.ylabel("Distance")
    plt.savefig("plot.jpg")
    plt.close()

def perform_algorithm(grade, data, size, pm, pc, limit):
    EA = Evolutionary_Algorithm(grade, data, size, pm, pc, limit)
    o_, x_ = EA.start_algorithm()
    return o_, x_

def generate_results(grade, data, size, pm, pc, limit):
    results = []
    for i in range(50):
        o, x = perform_algorithm(grade, data, size, pm, pc, limit)
        result = {"grade": o, "solution": x}
        results.append(result)
    return pd.DataFrame(results)

def best_result(data):
    result = data.loc[data["grade"].idxmin()].copy()
    df_result = pd.DataFrame(result).transpose().reset_index(drop=True)
    return df_result

def compare_parameters(data):
    results = {
        "population_size": [],
        "mutation_p": [],
        "crossover_p": [],
        "grade": []
    }

    population_size = np.arange(5, 50, 5)
    for size in population_size:
        o, x = perform_algorithm(evaluate_solution, data, size, 0.5, 0.5, 50)
        results["population_size"].append(size)
        results["mutation_p"].append(0.5)
        results["crossover_p"].append(0.5)
        results["grade"].append(o)

    mutation_p = np.arange(0.1, 1, 0.1)
    for pm in mutation_p:
        o, x = perform_algorithm(evaluate_solution, data, 20, pm, 0.5, 50)
        results["population_size"].append(20)
        results["mutation_p"].append(pm)
        results["crossover_p"].append(0.5)
        results["grade"].append(o)

    crossover_p = np.arange(0.1, 1, 0.1)
    for pc in crossover_p:
        o, x = perform_algorithm(evaluate_solution, data, 20, 0.5, pc, 50)
        results["population_size"].append(20)
        results["mutation_p"].append(0.5)
        results["crossover_p"].append(pc)
        results["grade"].append(o)

    data_res = pd.DataFrame(results)
    best_size = data_res.loc[data_res["grade"].idxmin(), "population_size"]
    best_mp = data_res.loc[data_res["grade"].idxmin(), "mutation_p"]
    best_cp = data_res.loc[data_res["grade"].idxmin(), "crossover_p"]

    return (best_size, best_mp, best_cp)

def avg_std_best_parameter(data):
    results = [compare_parameters(data) for i in range(10)]
    df = pd.DataFrame(results, columns=["population_size", "mutation_p", "crossover_p"])
    stats = df.agg(["mean", "std"]).transpose().round(2)
    return stats

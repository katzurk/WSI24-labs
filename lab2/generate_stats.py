import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from evolutionary_algorithm import Evolutionary_Algorithm
from solution_utils import *


def generate_plot(ea, filename):
    best, grade = zip(*(ea.development))
    plt.plot(np.arange(len(best)), grade, marker="o")
    plt.ylim(12000, max(grade)+250)
    plt.title("Best solution for each generation")
    plt.xlabel("Generation number")
    plt.ylabel("Distance")
    plt.savefig(f"{filename}.jpg")
    plt.close()

def perform_algorithm(grade, data, population_size, mutation_p, crossover_p, limit):
    EA = Evolutionary_Algorithm(grade, data, population_size, mutation_p, crossover_p, limit)
    o_, x_ = EA.start_algorithm()
    return o_, x_

def generate_results(grade, data, size, pm, pc, limit):
    results = []
    for i in range(10):
        o, x = perform_algorithm(grade, data, size, pm, pc, limit)
        result = {"grade": o, "solution": x}
        results.append(result)
    return pd.DataFrame(results)

def best_result(data):
    result = data.loc[data["grade"].idxmin()].copy()
    df_result = pd.DataFrame(result).transpose().reset_index(drop=True)
    return df_result

def compare_parameters(data, param_name, param_array, params):
    results = {
        "population_size": [],
        "mutation_p": [],
        "crossover_p": [],
        "grade": []
    }
    for param in param_array:
        params[param_name] = param
        o, x = perform_algorithm(evaluate_solution, data, **params)
        results["population_size"].append(params["population_size"])
        results["mutation_p"].append(params["mutation_p"])
        results["crossover_p"].append(params["crossover_p"])
        results["grade"].append(o)
    res_df = pd.DataFrame(results)
    return res_df

def run_tests(data):
    params = {"population_size": 20, "mutation_p": 0.5, "crossover_p": 0.5, "limit": 500}

    population_size = np.arange(10, 110, 10)
    population = compare_parameters(data, "population_size", population_size, params)

    mutation_p = np.arange(0.1, 1.1, 0.1)
    mutation = compare_parameters(data, "mutation_p", mutation_p, params)

    crossover_p = np.arange(0.1, 1.1, 0.1)
    crossover = compare_parameters(data, "crossover_p", crossover_p, params)

    best_size = population.loc[population["grade"].idxmin(), "population_size"]
    best_mp = mutation.loc[mutation["grade"].idxmin(), "mutation_p"]
    best_cp = crossover.loc[crossover["grade"].idxmin(), "crossover_p"]
    return (best_size, best_mp, best_cp)

def avg_std_best_parameter(data):
    results = [run_tests(data) for i in range(5)]
    df = pd.DataFrame(results, columns=["population_size", "mutation_p", "crossover_p"])
    stats = df.agg(["mean", "std"]).transpose().round(2)
    return stats

def crossover_prob_tests(data):
    params = {"population_size": 20, "mutation_p": 0.3, "crossover_p": 0.5, "limit": 500}
    crossover_p = np.arange(0.1, 1.1, 0.1)
    results = []
    for i in range(10):
        cp = compare_parameters(data, "crossover_p", crossover_p, params)
        results.append(cp)
    crossover_df = pd.concat(results, ignore_index=True)
    crossover_df = crossover_df.groupby("crossover_p")["grade"].agg(grade_mean="mean", grade_std="std")
    return crossover_df

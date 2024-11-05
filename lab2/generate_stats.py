import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from evolutionary_algorithm import Evolutionary_Algorithm


def generate_plot(ea):
    best, grade = zip(*(ea.development))
    plt.plot(np.arange(len(best)), grade, marker="o")
    plt.title("Best solution for each generation")
    plt.xlabel("Generation number")
    plt.ylabel("Distance")
    plt.show()

def perform_algorithm(grade, data, P0, pm, pc, limit):
    EA = Evolutionary_Algorithm(grade, data, P0, pm, pc, limit)
    o_, x_ = EA.start_algorithm()
    return o_, x_

def run_tests(grade, data, P0, pm, pc, limit):
    results = []
    for i in range(50):
        o, x = perform_algorithm(grade, data, P0, pm, pc, limit)
        results.append((o, x))
    return results

def best_result(results):
    sort_results = sorted(results)
    return sort_results[0]
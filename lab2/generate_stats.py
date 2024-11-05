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
    plt.show()

def perform_algorithm(grade, data, size, pm, pc, limit):
    EA = Evolutionary_Algorithm(grade, data, size, pm, pc, limit)
    o_, x_ = EA.start_algorithm()
    return o_, x_

def run_tests(grade, data, size, pm, pc, limit):
    results = []
    for i in range(50):
        o, x = perform_algorithm(grade, data, size, pm, pc, limit)
        results.append((o, x))
    return results

def best_result(results):
    sort_results = sorted(results)
    return sort_results[0]

def compare_parameters(data):
    population_size = np.arange(0, 50, 5)
    population_size = population_size[1:]
    best_g = float('inf')
    best_p1 = 0
    for size in population_size:
        o, x = perform_algorithm(evaluate_solution, data, size, 0.5, 0.5, 50)
        if o < best_g:
            best_g = o
            best_p1 = size

    mutation_p = np.arange(0.1, 1, 0.1)
    best_g = float('inf')
    best_p2 = 0
    for pm in mutation_p:
        o, x = perform_algorithm(evaluate_solution, data, 20, pm, 0.5, 50)
        if o < best_g:
            best_g = o
            best_p2 = pm

    crossover_p = np.arange(0.1, 1, 0.1)
    best_g = float('inf')
    best_p3 = 0
    for pc in crossover_p:
        o, x = perform_algorithm(evaluate_solution, data, 20, 0.5, pc, 50)
        if o < best_g:
            best_g = o
            best_p3 = pc

    return (best_p1, best_p2, best_p3)

def average_best_parameter(data):
    results = []
    for i in range(10):
        results.append(compare_parameters(data))
    s, m, c = zip(*results)
    return (sum(s)/len(s), sum(m)/len(m), sum(c)/len(c))
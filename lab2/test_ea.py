import pytest
import numpy as np
from evolutionary_algorithm import Evolutionary_Algorithm

def test_find_best():
    ea = Evolutionary_Algorithm(None, None, None, None, None, None)
    population = [[1,2,4,6], [3,2,6,8]]
    grades = [67, 45]
    best = ea.find_best(population, grades)
    assert best == (45, [3,2,6,8])

def test_reproduction():
    ea = Evolutionary_Algorithm(None, None, None, None, None, None)
    population = np.array([[1,2,4,6], [3,2,6,8]])
    grades = [67, 45]
    chosen = ea.reproduction(population, grades)
    case_1 = population[0] in chosen
    case_2 = population[1] in chosen
    assert case_1 and case_2 or case_1 and not case_2 or not case_1 and case_2
    assert len(chosen) == 2

# def test_mutating():
#     matrix = [
#         [1,1,1,1],
#         [1,1,1,1]
#     ]
#     ea = Evolutionary_Algorithm(None, matrix, None, 0.4, None, None)
#     population = np.array([[1,2,4,6], [3,2,6,8]])
#     M = ea.mutating(population)


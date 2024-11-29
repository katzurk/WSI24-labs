import pytest
import numpy as np
from evolutionary_algorithm import Evolutionary_Algorithm

def test_find_best():
    ea = Evolutionary_Algorithm(None, None, None, None, None, None)
    population = [[1,2,4,6], [3,2,6,8]]
    grades = [67, 45]
    best = ea.find_best(population, grades)
    assert best == (45, [3,2,6,8])

def test_selection():
    ea = Evolutionary_Algorithm(None, None, None, None, None, None)
    population = [[1,2,4,6], [3,2,6,8]]
    grades = [67, 45]
    chosen = ea.selection(population, grades)
    case_1 = population[0] in chosen
    case_2 = population[1] in chosen
    assert case_1 and case_2 or case_1 and not case_2 or not case_1 and case_2
    assert len(chosen) == 2

def test_mutation():
    ea = Evolutionary_Algorithm(None, None, None, 0.4, None, None)
    population = [[1,2,4,6,7,3,5,9], [3,2,6,8]]
    M = ea._mutation(population[0])
    assert sorted(population[0]) == sorted(M)

def test_crossover():
    ea = Evolutionary_Algorithm(None, None, None, 0.4, None, None)
    population = [[1,2,3,6], [3,2,6,8]]
    child = ea._crossover(population[0], population[1])
    assert len(child) == len(population[0])

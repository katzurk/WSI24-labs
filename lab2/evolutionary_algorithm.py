from solution_utils import *
import numpy as np


class Evolutionary_Algorithm():
    def __init__(g, matrix, P0, pm, pc, limit):
        self.grade = g
        self.matrix = matrix
        self.population = P0 # vector of specimens
        self.pm = pm
        self.pc = pc
        self.limit = limit

    def start_algorithm(self):
        t = 0
        o = self.grade(matrix, self.population)
        o_, x_ = find_best(self.population, o)
        while not stop_condition(t, self.population):
            R = reproduction(o)
            M = genetic_operations(R)
            o_m = self.grade(matrix, M)
            o_t, x_t = find_best(M, o_m)
            if o_t < o_:
                o_, x_ = o_t, x_t
            self.population, o = succession(M, o, o_m)
            t += 1


    def find_best(self, population, grades):
        sorted_p = sorted(zip(grades, population))
        return sorted_p[0]

    def stop_condition(self, iteration, current_population):
        if iteration < limit:
            return 0
        return 1

    def reproduction(self, grades):
        # roulette selection
        ps = []
        grade_sum = sum(grades)
        for grade in grades:
            ps_i = grade / grade_sum
        chosen = np.random_choice(self.population, len(self.population), ps)
        return chosen

    def genetic_operations(self, population):
        return 0

    def mutating(self, population):
        return 0

    def crossing(self, population):
        return 0

    def succession(self, m_population, grades, m_grades):
        return 0

from solution_utils import *
import numpy as np


class Evolutionary_Algorithm:
    def __init__(self, g, matrix, size, pm, pc, limit):
        self.grade = g  # grading function
        self.matrix = matrix
        self.population = [
            generate_solution(self.matrix) for i in range(size)
        ]  # vector of solutions, starting population
        self.pm = pm  # probability of mutation
        self.pc = pc  # probability of crossover
        self.limit = limit  # generation count limit
        self.development = []  # best solution of each generation

    def start_algorithm(self):
        t = 0
        o = self.evaluate(self.population)
        o_, x_ = self.find_best(self.population, o)
        while not self.stop_condition(t, self.population):
            R = self.selection(self.population, o)
            M = self.genetic_operations(R)
            o_m = self.evaluate(M)
            o_t, x_t = self.find_best(M, o_m)
            if o_t < o_:
                o_, x_ = o_t, x_t
            self.population, o = self.succession(M, o, o_m)
            self.development.append((x_, o_))
            t += 1
        return (o_, x_)

    def evaluate(self, population):
        grades = [self.grade(self.matrix, sol) for sol in population]
        return grades

    def find_best(self, population, grades):
        sorted_p = sorted(zip(grades, population))
        return sorted_p[0]

    def stop_condition(self, iteration, current_population):
        if iteration < self.limit:
            return 0
        return 1

    def selection(self, population, grades):
        # roulette selection
        grades_m = [1 / grade for grade in grades]  # less distance - more probable
        grade_sum = sum(grades_m)
        ps = [grade / grade_sum for grade in grades_m]
        index = np.arange(len(population))
        chosen_id = np.random.choice(index, size=len(population), p=ps, replace=True)
        chosen = np.array(population)[chosen_id]
        return chosen.tolist()

    def genetic_operations(self, population):
        C = self.crossover(population)
        M = self.mutation(C)
        return M

    def mutation(self, population):
        mutation = []
        for solution in population:
            if np.random.rand() > self.pm:
                mutation.append(solution)
                continue
            m_solution = self._mutation(solution)
            validate_solution(self.matrix, m_solution)
            mutation.append(m_solution)
        return mutation

    def _mutation(self, solution):
        # swap mutation
        i, j = np.random.randint(1, len(solution) - 1, 2)
        while i == j:
            j = np.random.randint(1, len(solution) - 1)
        solution[i], solution[j] = solution[j], solution[i]
        return solution

    def crossover(self, population):
        crossover = []
        for idx, solution in enumerate(population):
            if np.random.rand() > self.pc:
                crossover.append(solution)
                continue
            parent_1 = solution
            parent_2_id = np.random.choice(
                [i for i in range(len(population)) if i != idx]
            )
            parent_2 = population[parent_2_id]
            child = self._crossover(parent_1, parent_2)
            validate_solution(self.matrix, child)
            crossover.append(child)
        return crossover

    def _crossover(self, parent_1, parent_2):
        # order crossover (OX)
        i, j = np.random.randint(0, len(parent_1), 2)
        while i == j:
            j = np.random.randint(0, len(parent_1))
        child = [None] * len(parent_1)
        child[i:j] = parent_1[i:j]
        idx = 0
        for element in parent_2:
            if element not in child:
                while child[idx] is not None:
                    idx += 1
                child[idx] = element
        return child

    def succession(self, m_population, grades, m_grades):
        return m_population, m_grades

from solution_utils import *
import numpy as np


class Evolutionary_Algorithm:
    def __init__(self, g, matrix, P0, sigma, pc, limit):
        self.grade = g
        self.matrix = matrix
        self.population = P0  # vector of specimens
        self.sigma = sigma
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

    def reproduction(self, population, grades):
        # roulette selection
        ps = []
        grade_sum = sum(grades)
        for grade in grades:
            ps_i = grade / grade_sum
            ps.append(ps_i)
        index = np.arange(len(population))
        chosen_id = np.random.choice(index, size=len(population), p=ps)
        chosen = population[chosen_id]
        return chosen

    def genetic_operations(self, population):
        C = crossing(population)
        M = mutating(C)
        return M

    def mutating(self, population):
        # inversion or rotation
        mutating = []
        for solution in population:
            rnd = np.random.rand()
            if rnd < self.sigma:
                mutating.append(solution)
                continue
            m_solution = mutate(solution)
            if validate_solution(self.matrix, m_solution):
                mutating.append(m_solution)
        return mutating

    def mutate(self, solution):
        temp_solution = solution[1:-1]
        i = np.random.randint(0, len(temp_solution)-2)
        j = np.random.randint(i+2, len(temp_solution))
        part_list = temp_solution[i:j]
        part_list = part_list[::-1]
        m_solution = solution[:i+1] + part_list + solution[j+1:]
        return solution

    def crossing(self, population):
        crossing = []
        for s_id, solution in enumerate(population):
            weights = np.random.choice([0, 1], size=len(solution), p=[self.pc, 1-self.pc])
            parent_1 = solution
            parent_2_id = np.random.choice([i for i in range(len(population)) if i!=s_id])
            parent_2 = population[parent_2_id]
            child = np.where(weights, parent_1, parent_2)
            # if validate_solution(self.matrix, child):
            crossing.append(child)
        return crossing

    def succession(self, m_population, grades, m_grades):
        return m_population, m_grades

"""
Implementation of the NSGAII MOEA
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT. 
"""
from hklearn_genetic.Problems.IProblem import IProblem
from .utils import PlotUtils, ProblemUtils
import numpy as np
import copy

PLAY = True

class NSGAII:
    def __init__(self, max_iter : int = 5000, tournament_type : int = 0, tournament_size = 2):
        self.max_iter = max_iter
        self.k_tournament = tournament_size
        self.tournament_type = tournament_type

    def evolve(self, problem : IProblem, n_individuals : int, debug : bool = False):
        self.debug = debug
        #Initialize population
        its = 0
        parent_pop = problem.populate(n_individuals)
        while its < self.max_iter: 
            child_pop = problem.generate(parent_pop)
            population = copy.deepcopy(parent_pop) + child_pop
            problem.evaluate(population)
            parent_pop = self._select(population, n_individuals)
            its+=1

        return self._non_dominated_sorting(population)[0]
        # if len(solutions) == 0:
        #     return (self.pop[-1].phenotype, -self.pop[-1].fitness_metric)
        # else:
        #     return solutions

    def _non_dominated_sorting(self, population : list):
        fronts = [[]]
        for i in range(len(population)):
            population[i].dominated = []
            population[i].domination_counter = 0

            for j in range(len(population)):
                if i != j:
                    if population[i].dominates(population[j]):
                        population[i].dominated.append(population[j])
                    elif population[j].dominates(population[i]):
                        population[i].domination_counter += 1
            if population[i].domination_counter == 0:
                fronts[0] += [population[i]]
                population[i].rank = 1
        #Initialize front counter
        i = 0
        while len(fronts[i]) != 0:
            next_front_members = []
            for p in fronts[i]:
                for q in p.dominated:
                    q.domination_counter -= 1
                    if q.domination_counter == 0:
                        q.rank = i + 1
                        next_front_members += [q]
            fronts.append(next_front_members)
            i+=1
                
        fronts.pop()
        return fronts

    def _get_crowding_distance(self, front):
        M = len(front[0].fitness_metric)
        min_max = []
        for i in range(M):
            max_val = float('-inf')
            min_val = float('inf')
            for p in front:
                if max_val < p.fitness_metric[i]:
                    max_val = p.fitness_metric[i]
                if min_val > p.fitness_metric[i]:
                    min_val = p.fitness_metric[i]
            min_max += [(min_val, max_val)]

        for p in front:
            p.crowding_distance = 0
        for i in range(M):
            front.sort(key = lambda x : x.fitness_metric[i])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (front[j + 1].fitness_metric[i] - front[j - 1].fitness_metric[i])/(min_max[i][1] - min_max[i][0]) 
        return front


    def _get_roulette_probs(self, n_individuals : int):
        return np.random.uniform(size = (1, n_individuals))

    def _select(self, population : list, n_individuals : int):
        """Evaluates the population of solutions based on the problem and 
        applies the chosen selection operator over the evaluated proposed solutions
        Parameters
        ----------
        problem : IProblem
            problem object wich implements the IProblem interface
        population : list
            list of objects, each representing a proposed solution
        """
        fronts = self._non_dominated_sorting(population)
        if self.debug:
            front_mats = []
            for front in fronts:
                front_mat = ProblemUtils._to_matrix_fitness(front)
                front_mats += [front_mat]
            PlotUtils.plot_mat_3d(front_mats, "Fronts", "", play=PLAY)
        parent_pop = []
        for front in fronts:

            if len(front) + len(parent_pop) > n_individuals:
                self._get_crowding_distance(front)
                front.sort(key = lambda x : x.crowding_distance, reverse = True)
                parent_pop += front[0: n_individuals - len(parent_pop)]
                break
            elif len(front) + len(parent_pop) == n_individuals:
                parent_pop += front
                break
            else:
                parent_pop += front

        selected_parent_pop = []
        t = 0
        #with replacement
        if self.tournament_type == 1:
            
            while t < len(parent_pop):
                tournament_contestants = np.random.permutation(len(parent_pop))[0:self.k_tournament]
                greatest_score_so_far = float('inf')
                tournament_winner = None
                for contestant in tournament_contestants:
                    if parent_pop[contestant].rank < greatest_score_so_far:
                        greatest_score_so_far = parent_pop[contestant].rank
                        tournament_winner = copy.deepcopy(parent_pop[contestant])
                selected_parent_pop += [tournament_winner]
                t+=1
        #without replacement
        elif self.tournament_type == 0:
            while t < len(parent_pop):
                permutation = np.random.permutation(len(parent_pop))
                i = 0
                tournament_winner = None
                while i < len(permutation) and t < len(parent_pop):
                    greatest_score_so_far = float('inf')
                    for j in range(i,min(i + self.k_tournament, len(parent_pop))):
                        if parent_pop[permutation[j]].rank < greatest_score_so_far:
                            greatest_score_so_far = parent_pop[j].rank
                            tournament_winner = copy.deepcopy(parent_pop[permutation[j]])
                    selected_parent_pop += [tournament_winner]
                    t+=1
                    i+=self.k_tournament
        return selected_parent_pop

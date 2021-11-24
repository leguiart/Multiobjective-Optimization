
"""
The generic evolutionary algorithm
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT. 
"""
from hklearn_genetic.Problems.IProblem import IProblem
import numpy as np
import random
import math
import copy

class EvolutionaryAlgorithm:
    """
    The generic evolutionary algorithm is designed to be the context class to any particular implementation.
    We follow a strategy pattern in which each strategy (problem) provides its own definition of the IProblem interface.
    ...

    Attributes
    ----------
    pc : float
        Crossover probability
    pm : float
        Mutation probability
    max_iter : int
        Maximum number of iterations (generations) to perform 
    selection : string
        Selection method, one of - proportional or tournament
    elitism : float
        Proportion of best performing individuals of population to be kept for the next generation
    tournament_type : int
        Tournament variant, one of - without replacement (0) or with replacement (1)
        Based on the description given in - https://wpmedia.wolfram.com/uploads/sites/13/2018/02/03-5-5.pdf
        
    Methods
    -------
    evolve(problem, n_individuals)
        Applies genetic algorithm until a stop criteria is met.
        Returns list of proposed solutions that meet the stop criteria.
    """
    def __init__(self, max_iter : int = 5000, selection : str = "proportional" , elitism : float = 0., tournament_type : int = 0):
        """
        Parameters
        ----------
        pc : float, optional
            Crossover probability (default is 0.6)
        pm : float, optional
            Mutation probability (default is 0.1)
        max_iter : int, optional
            Maximum number of iterations (generations) to perform (default is 5000)
        selection : string, optional
            Selection method, one of - proportional or tournament (default is "proportional")
        elitism : float, optional
            Proportion of best performing individuals of population to be kept for the next generation (default is 0)
        tournament_type : int, optional
            Tournament variant, one of - without replacement (0) or with replacement (1) (default is 0)
            Based on the description given in - https://wpmedia.wolfram.com/uploads/sites/13/2018/02/03-5-5.pdf
        """
        self.max_iter = max_iter
        self.elitism = elitism
        self.k_tournament = 2
        self.tournament_type = tournament_type
        self.selection = selection

        
    
    def evolve(self, problem : IProblem, n_individuals : int):
        """Applies the simple ga in order to evolve a population of proposed solutions
        Parameters
        ----------
        problem : IProblem
            problem object wich implements de IProblem interface
        n_individuals : int
            size of the population of proposed solutions
        """
        its = 0
        self.elitism_num = math.floor(self.elitism*n_individuals)
        self.pop = problem.populate(n_individuals)
        while not problem.stop_criteria(self.pop) and its < self.max_iter: 
            self.pop = self._select(problem, self.pop)
            self.pop = problem.generate(self.pop)
            its+=1

        solutions = problem.extract_solutions()
        if len(solutions) == 0:
            return (self.pop[-1].phenotype, -self.pop[-1].fitness_metric)
        else:
            return solutions


    def _get_roulette_probs(self, n_individuals : int):
        return np.random.uniform(size = (1, n_individuals))

    def _select(self, problem : IProblem, population : list):
        """Evaluates the population of solutions based on the problem and 
        applies the chosen selection operator over the evaluated proposed solutions
        Parameters
        ----------
        problem : IProblem
            problem object wich implements the IProblem interface
        population : list
            list of objects, each representing a proposed solution
        """
        population = problem.evaluate(population)
        population.sort(key = lambda x : x.fitness_metric)
        fitness_metrics = [individual.fitness_metric for individual in population]

        if self.selection == "proportional":
            prob_sel = np.array(fitness_metrics)/sum(fitness_metrics)
            c_prob_sel = np.cumsum(prob_sel)
            probs = self.get_roulette_probs(len(population) - self.elitism_num)

            for j, prob in enumerate(probs[0, :]):
                i = np.searchsorted(c_prob_sel, prob)
                population[j] = copy.deepcopy(population[i])

        elif self.selection == "tournament":
            t = 0
            #with replacement
            if self.tournament_type == 1:
                
                while t < len(population) - self.elitism_num:
                    tournament_contestants = np.random.permutation(len(population))[0:self.k_tournament]
                    greatest_score_so_far = float('-inf')
                    for contestant in tournament_contestants:
                        if population[contestant].fitness_metric > greatest_score_so_far:
                            greatest_score_so_far = population[contestant].fitness_metric
                            population[t] = copy.deepcopy(population[contestant])
                    t+=1
            #without replacement
            elif self.tournament_type == 0:
                while t < len(population) - self.elitism_num:
                    permutation = np.random.permutation(len(population))
                    i = 0
                    while i < len(permutation) and t < len(population) - self.elitism_num:
                        greatest_score_so_far = float('-inf')
                        for j in range(i,min(i + self.k_tournament, len(population))):
                            if population[permutation[j]].fitness_metric > greatest_score_so_far:
                                greatest_score_so_far = population[j].fitness_metric
                                population[t] = copy.deepcopy(population[permutation[j]])
                        t+=1
                        i+=self.k_tournament
        elif self.selection == "sus":
            prob = np.array(fitness_metrics).sum()/(len(population) - self.elitism_num)
            start = random.random()
            pointers = [start*prob + i*prob for i in range(len(population) - self.elitism_num)]
            c_fitness = np.cumsum(np.array(fitness_metrics))
            for j, p in enumerate(pointers):
                i = np.searchsorted(c_fitness, p)
                population[j] = copy.deepcopy(population[i])
        return population

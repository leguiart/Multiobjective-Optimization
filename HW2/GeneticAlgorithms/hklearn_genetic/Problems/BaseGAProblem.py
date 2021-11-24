import abc
from ..Evaluators.IEvaluator import IEvaluator
import numpy as np
import math
from hklearn_genetic.Problems.IProblem import IProblem
from ..utils import ProblemUtils
import copy


class _BaseGAProblem(IProblem):
    """
    Abstract base class for all problems to be solved under a GA approach.
    We follow a template pattern.
    ...

    Attributes
    ----------
    ... We list the attributes of the class 
    e.g.
    name : type
        Description 
    Methods
    -------
    ... List of public methods of the form:
    function(args*)
        Description
    """
    #def __init__(self, evaluator : IEvaluator, thresh, pc : float = 0.6, pm : float = 0.1, elitism : float = 0., analytics = None):
    def __init__(self, params):
        self.evaluator = params["evaluator"]
        self.thresh = params["thresh"]
        self.pc = params["pc"] if "pc" in params else 0.6
        self.pm = params["pm"] if "pm" in params else 0.1
        self.elitism = params["elitism"]  if "elitism" in params else 0.
        self.solutions = []
        self.analytics = params["analytics"] if "analytics" in params else None

    def generate(self, pop : list) -> list:
        elitism_num = math.floor(self.elitism*len(pop))
        pop_c = copy.deepcopy(pop)
        children_crossover = self._crossover(pop_c, self.pc, elitism_num)
        children = self._mutate(children_crossover, self.pm, elitism_num)
        return children
    
    def evaluate(self, X : list) -> list:
        """Applies an evaluation function over an encoded representation of the population
        by first decoding the genotype, then filling the phenotype representation and finally evaluating
        that phenotype
        Parameters
        ----------
        X : list
            List of individuals containing their respective genotype
        """
        X_decoded = self._decode(X)
        self.evaluator.evaluate(X_decoded)
        if self.analytics:
            self.analytics.gather_analytics(X_decoded)

    def _initialize_from_mat(self, X_mat):
        individuals = []
        for i in range(len(X_mat)):
            individual = Individual()
            individual.genotype = X_mat[i].copy()
            individuals+=[individual]
        return individuals

    def stop_criteria(self, X_eval : list) -> bool:
        """Evaluates if the stop criteria for a population has been met  
        Parameters
        ----------
        X_eval : list
            List of evaluated individuals representing the population of proposed solutions
        """
        self.solutions = list(filter(lambda x : x.fitness_metric >= self.thresh, X_eval))
        if len(self.solutions) != 0:
            # if self.analytics:
            #     self.analytics.gather_analytics(X_eval)
            return True

    def extract_solutions(self):
        if len(self.solutions) != 0:
            return [(sol.phenotype, -sol.fitness_metric) for sol in self.solutions]
        else:
            return []

    @abc.abstractmethod
    def _decode(self, X):
        pass
    
    # @abc.abstractmethod
    # def _gather_analytics(self, X_eval):
    #     pass

    @abc.abstractmethod
    def _get_parameters(self):
        pass

    @abc.abstractmethod
    def _crossover(self, X, pc, elitism):
        pass

    @abc.abstractmethod
    def _mutate(self, X, pc, elitism):
        pass


class Individual:
    def __init__(self):
        self.genotype = []
        self.phenotype = []
        self.fitness_metric = 0
        self.rank = -1
        self.dominated = []
        self.domination_counter = 0
        self.crowding_distance = 0
        self.reference_line = []
        self.distance = 0

    def copy(self):
        cpy = Individual()
        cpy.genotype = self.genotype.copy()
        cpy.phenotype = self.phenotype.copy()
        cpy.fitness_metric = self.fitness_metric
        return cpy

    def dominates(self, x):
        return (self.fitness_metric <= x.fitness_metric).all() and (self.fitness_metric != x.fitness_metric).any()

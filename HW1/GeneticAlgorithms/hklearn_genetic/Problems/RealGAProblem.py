"""
Real number problems structure to be used in genetic algorithm approaches based on the breeder genetic algorithm.
Defines a problem which genotype is codified in the domain of real numbers.
Implements SBX crossover and polynomial mutation
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""
import numpy as np
from numpy.random.mtrand import beta
from hklearn_genetic.Problems.BaseGAProblem import _BaseGAProblem
from ..Evaluators.IEvaluator import IEvaluator
from ..utils import ProblemUtils

class _BaseRealGAProblem(_BaseGAProblem):
    """
    Base class for all problems codified with real number vectors.
    ...

    Attributes
    ----------
    evaluator : Evaluator
        Object defining the way to evaluate the phenotypes
    n_dim : float
        Dimensions of the genotype vectors
    thresh : float
        Fitness function value threshold to decide a finishing criteria
    bounds : tuple (float, float)
        Restriction on the domain of the problem, equal for each dimension
    rang_param : float
        Fixed rate of the mutation range
        
    Methods
    -------
    populate(n_individuals)
        Fill out a real number matrix of dimensions n_individuals X n_dim with random values in (bounds[0], bounds[1])
        representing the population genotype pool
    stop_criteria(self, X_eval)
        Evaluates if the stop criteria for a population has been met  
    crossover(X, pc, elitism_num)
        Applies crossover operator over a real number matrix of dimensions n_individuals X n_dim
    mutate(X, pm, elitism_num)
        Applies mutation operator over a real number matrix of dimensions n_individuals X n_dim
    evaluate(X)
        Applies an evaluation function over an encoded representation of the population
        by first decoding the genotype, filling the phenotype representation and then evaluating
        that phenotype
    """
    def __init__(self, evaluator : IEvaluator, thresh, bounds : tuple, pc : float = 0.6, pm : float = 0.1, elitism : float = 0., rang_param : float = 0.1, n_dim : int= 2, analytics = None):
        """
        Parameters
        ----------
        evaluator : Evaluator
            Object defining the way to evaluate the phenotypes
        n_dim : int, optional
            Dimensions of the genotype vectors (default is 2)
        thresh : float
            Fitness function value threshold to decide a finishing criteria
        bounds : tuple (float, float)
            Restriction on the domain of the problem, equal for each dimension
        rang_param : float, optional
            Fixed rate of the mutation range (default is 0.1)
        """
        super().__init__(evaluator, thresh, pc=pc, pm=pm, elitism=elitism, analytics=analytics)
        self.n_dim = n_dim
        self.bounds = bounds
        self.rang_param = rang_param
        self.generation_counter = 1

    def populate(self, n_individuals : int) -> list:
        """Fill out a real number matrix of dimensions n_individuals X n_dim with random values in (bounds[0], bounds[1])
        representing the population genotype pool
        n_individuals : int
            Size of the population
        """
        X_mat = np.random.uniform(self.bounds[0], self.bounds[1], size = (n_individuals, self.n_dim))
        return self._initialize_from_mat(X_mat)

    def _decode(self, X):
        for i in range(len(X)):
            X[i].phenotype = X[i].genotype
        return X
    
    # @abc.abstractmethod
    # def _gather_analytics(self, X_eval):
    #     pass

    def _get_crossover_probs(self, n_cross):
        return np.random.rand(1 , n_cross)[0,:]

    def _crossover(self, X, pc, elitism_num):
        """Applies SBX crossover operator over a real number matrix of dimensions n_individuals X n_dim
        Parameters
        ----------
        X : list
            Genotype matrix of dimensions n_individuals X n_dim
        pc : float
            Crossover probability
        elitism_num : int
            Number of individuals from the last rows to be kept without modification
        """
        X_mat = ProblemUtils._to_matrix(X)
        n_cross = (X_mat.shape[0] - elitism_num) // 2
        prob_cross = self._get_crossover_probs(n_cross)
        n_c = 1
        for i, p in enumerate(prob_cross):
            if p <= pc:
                u = np.random.uniform(0.,1.)
                beta = (2*u)**(1/(1+n_c)) if u <= 0.5 else (1/(2*(1-u)))**(1/(1+n_c))
                X_mat[2*i,:] = 0.5 * ((1 + beta)*X_mat[2*i,:] + (1 - beta)*X_mat[2*i + 1, :])
                X_mat[2*i + 1,:] = 0.5 * ((1 - beta)*X_mat[2*i,:] + (1 + beta)*X_mat[2*i + 1, :])
                X_mat[2*i,:] = np.clip(X_mat[2*i,:], self.bounds[0], self.bounds[1])
                X_mat[2*i + 1,:] = np.clip(X_mat[2*i + 1,:], self.bounds[0], self.bounds[1])
        return ProblemUtils._to_genotypes(X, X_mat)

    def _get_mutation(self, shape):
        return np.random.uniform(size = shape)
    
    def _mutate(self, X, pm, elitism_num):
        """Applies polynomial mutation over a real number matrix of dimensions n_individuals X n_dim
        Parameters
        ----------
        X : list
            Genotype matrix of dimensions n_individuals X n_dim
        pm : float
            Mutation probability
        elitism_num : int
            Number of individuals from the last rows to be kept without modification
        """
        X_mat = ProblemUtils._to_matrix(X)
        delta_max = self.bounds[1] - self.bounds[0]
        #We create a matrix of random numbers which will tell us if the alele of a given individual will be mutated or not
        mutate_m = self._get_mutation((X_mat.shape[0], X_mat.shape[1]))
        #We obtain all the values of u for each alele of each individual
        delta_qs = self._get_mutation((X_mat.shape[0], X_mat.shape[1]))
        #Each element of the mutate_m matrix which is less than or equal to pm will be set to one
        mutate_m[mutate_m <= pm] = 1.
        mutate_m[mutate_m < 1.] = 0.
        #We assign the delta_qs values based on the u values
        n_m = 100 + self.generation_counter
        u_s = delta_qs.copy()
        delta_qs[u_s < .5] = (2*u_s[u_s < .5])**(1/(1+n_m))-1.
        delta_qs[u_s >= .5] = 1. - (2*(1-u_s[u_s >= .5]))**(1/(1+n_m))

        for i in range(X_mat.shape[0] - elitism_num):
            X_mat[i, :] = X_mat[i, :] + mutate_m[i, :] * delta_qs[i, :] * delta_max
            X_mat[i, :] = np.clip(X_mat[i, :], self.bounds[0], self.bounds[1])
        return ProblemUtils._to_genotypes(X, X_mat)
    

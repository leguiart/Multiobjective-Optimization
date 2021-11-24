"""
Real number problems structure to be used in genetic algorithm approaches based on the breeder genetic algorithm.
Defines a problem which genotype is codified in the domain of real numbers.
Based on the BGA algorithm for continous parameter optimization (https://ieeexplore.ieee.org/document/6792992).
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""
import numpy as np
from hklearn_genetic.Problems.BaseGAProblem import _BaseGAProblem
from ..Evaluators.IEvaluator import IEvaluator
from ..utils import ProblemUtils

class _BaseRealBGAProblem(_BaseGAProblem):
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
    def __init__(self, evaluator : IEvaluator, thresh, bounds : tuple, pc : float = 0.6, pm : float = 0.1, elitism : float = 0., rang_param : float = 0.1, n_dim : int= 2):
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
        super().__init__(evaluator, thresh, pc=pc, pm=pm, elitism=elitism)
        self.n_dim = n_dim
        self.bounds = bounds
        self.rang_param = rang_param

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

    def _get_crossover_points(self, length):
        return np.random.uniform(low = -.25 , high = 1.25, size = length)

    def _crossover(self, X, pc, elitism_num):
        """Applies crossover operator over a real number matrix of dimensions n_individuals X n_dim
        as described here: https://ieeexplore.ieee.org/document/6792992
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
        #Extended intermediate recombination
        for i, p in enumerate(prob_cross):
            if p <= pc:
                alphas = self._get_crossover_points(X_mat.shape[1])
                X_mat[2*i,:] += alphas * (X_mat[2*i + 1, :] - X_mat[2*i,:])
                X_mat[2*i + 1,:] += alphas * (X_mat[2*i,:] - X_mat[2*i + 1, :])
                X_mat[2*i,:] = np.clip(X_mat[2*i,:], self.bounds[0], self.bounds[1])
                X_mat[2*i + 1,:] = np.clip(X_mat[2*i + 1,:], self.bounds[0], self.bounds[1])
        return ProblemUtils._to_genotypes(X, X_mat)

    def _get_mutation(self, shape):
        return np.random.uniform(size = shape)
    
    def _mutate(self, X, pm, elitism_num):
        """Applies mutation operator over a real number matrix of dimensions n_individuals X n_dim
        as described here: https://ieeexplore.ieee.org/document/6792992
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
        rang = (self.bounds[1] - self.bounds[0])*self.rang_param
        mutate_m = self._get_mutation((X_mat.shape[0], X_mat.shape[1]))
        
        mutate_plus_minus = self._get_mutation((X_mat.shape[0], X_mat.shape[1]))

        mutate_m[mutate_m <= pm] = 1.
        mutate_m[mutate_m < 1.] = 0.
        mutate_plus_minus[mutate_plus_minus <= .5] = 1.0
        mutate_plus_minus[mutate_plus_minus != 1.0] = -1.0

        for i in range(X_mat.shape[0] - elitism_num):
            mutate_delta = self._get_mutation((X_mat.shape[1], X_mat.shape[1]))
            mutate_delta[mutate_delta <= 1./self.n_dim] = 1.
            mutate_delta[mutate_delta < 1.] = 0.
            deltas = (mutate_delta @ (2**-np.arange(self.n_dim, dtype = np.float64)[:, np.newaxis])).T
            X_mat[i, :] = X_mat[i, :] + mutate_m[i, :] * mutate_plus_minus[i, :] * rang * deltas
            X_mat[i, :] = np.clip(X_mat[i, :], self.bounds[0], self.bounds[1])
        return ProblemUtils._to_genotypes(X, X_mat)
    

import abc
from hklearn_genetic.utils import ProblemUtils
import numpy as np
import math
import random
import itertools as it
from hklearn_genetic.Problems.BaseGAProblem import _BaseGAProblem

class _BaseBinaryGAProblem(_BaseGAProblem):
    def __init__(self, evaluator, thresh, bounds, pc = 0.6, pm = 0.1, elitism = 0, n_dim = 2, n_prec = 4):
        super().__init__(evaluator, thresh, pc=pc, pm=pm, elitism=elitism)
        self.bounds = bounds
        self.n_dim = n_dim
        self.gene_length = math.ceil(math.log2((self.bounds[1] - self.bounds[0])*10**n_prec))

    def populate(self, n_individuals : int) -> list:
        X_mat = np.random.randint(2, size = (n_individuals, self.gene_length*self.n_dim))
        return self._initialize_from_mat(X_mat)

    def _decode(self, X):
        decoded_rep = np.zeros((len(X), self.n_dim))
        for i in range(self.n_dim):
            decoded_rep[:,i] = (X[:, i*self.gene_length : (i + 1)*self.gene_length]@(2**np.arange(X[:, i*self.gene_length : (i + 1)*self.gene_length].shape[1], dtype = np.float64)[::-1][:, np.newaxis])).T
        decoded_rep = self.bounds[0] + decoded_rep*(self.bounds[1] - self.bounds[0])/(2**self.gene_length - 1)
        for i in range(len(X)):
            X[i].phenotype = decoded_rep[i]
        return X
        
    
    def _get_crossover_probs(self, n_cross):
        return np.random.rand(1 , n_cross)[0,:]

    def _get_crossover_points(self, length):
        return np.random.randint(0, length)

    def _crossover(self, X, pc, elitism_num):
        X_mat = ProblemUtils._to_matrix(X)
        n_cross = (X_mat.shape[0] - elitism_num) // 2
        prob_cross = self._get_crossover_probs(n_cross)
        for i, p in enumerate(prob_cross):
            if p <= pc:
                cross_point = self._get_crossover_points(X_mat.shape[1] - 1)
                son1 = X_mat[2*i + elitism_num,:].copy()
                son2 = X_mat[2*i + 1 + elitism_num, :].copy()
                son1[cross_point : X_mat.shape[1]] = X_mat[2*i + 1 + elitism_num, cross_point : X_mat.shape[1]].copy()
                son2[cross_point : X_mat.shape[1]] = X_mat[2*i + elitism_num, cross_point : X_mat.shape[1]].copy()
                X_mat[2*i + elitism_num,:] = son1
                X_mat[2*i + 1 + elitism_num,:] = son2
        return ProblemUtils._to_genotypes(X, X_mat)

    def _get_mutation(self, shape):
        return np.random.uniform(size = shape)

    def _mutate(self, X, pm, elitism_num):
        X_mat = ProblemUtils._to_matrix(X)
        mutate_m = self._get_mutation((X_mat.shape[0], X_mat.shape[1]))
        mutate_m = mutate_m <= pm
        X_mat_bit = X_mat == 1       
        X_mat[0 : X_mat.shape[0] - elitism_num, :] = np.logical_X_mator(X_mat_bit, mutate_m)[0 : X_mat.shape[0] - elitism_num, :]
        X_mat = X_mat.astype(int)
        return ProblemUtils._to_genotypes(X, X_mat)

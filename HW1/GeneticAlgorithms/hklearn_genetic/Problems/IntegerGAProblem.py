import abc
from hklearn_genetic.utils import ProblemUtils
import numpy as np
import math
import random
import itertools as it
from hklearn_genetic.Problems.BaseGAProblem import _BaseGAProblem

class _BaseIntegerGAProblem(_BaseGAProblem):
    def __init__(self, evaluator, thresh, pc = 0.6, pm = 0.1, elitism = 0., n_dim = 2):
        super().__init__(evaluator, thresh, pc=pc, pm=pm, elitism=elitism)
        self.n_dim = n_dim

    def populate(self, n_individuals : int) -> list:
        X_mat = np.random.randint(self.n_dim, size = (n_individuals, self.n_dim))
        return self._initialize_from_mat(X_mat)

    def _decode(self, X):
        for i in range(len(X)):
            X[i].phenotype = X[i].genotype
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
        mutate_m = self._get_mutation((X.shape[0], 1))
        mutate_m = mutate_m <= pm
        X_mat = ProblemUtils._to_matrix(X)
        for i in range(X_mat.shape[0] - elitism_num):
            if mutate_m[i]:
                indices = np.random.permutation(X_mat.shape[1])[0 : 2]
                X_mat[i,indices[0]], X_mat[i, indices[1]] = X_mat[i, indices[1]], X_mat[i, indices[0]]
        return ProblemUtils._to_genotypes(X, X_mat)

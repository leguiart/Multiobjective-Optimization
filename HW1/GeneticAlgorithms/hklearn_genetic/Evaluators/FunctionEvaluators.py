from hklearn_genetic.utils import ProblemUtils
from hklearn_genetic.Evaluators.IEvaluator import IEvaluator
import numpy as np

class BaseRastrigin(IEvaluator):
    def __init__(self, rank : float = 100., dim = 2):
        self.rank = rank
        self.n_dim = dim

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        X_eval = self.rank - (10.*self.n_dim + np.sum(X_mat**2 - 10.*np.cos(2.*np.pi*X_mat), axis = 1))
        return ProblemUtils._to_evaluated(X, X_eval)

class BaseBeale(IEvaluator):
    def __init__(self, rank : float = 150000.):
        self.rank = rank

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        first_term = (1.5 - X_mat[:, 0] + X_mat[:, 0]*X_mat[:, 1])**2
        second_term = (2.25 - X_mat[:, 0] + X_mat[:, 0]*(X_mat[:, 1]**2))**2
        third_term = (2.625 - X_mat[:, 0] + X_mat[:, 0]*(X_mat[:, 1]**3))**2
        X_eval = self.rank - (first_term + second_term + third_term)
        return ProblemUtils._to_evaluated(X, X_eval)


class BaseHimmelblau(IEvaluator):
    def __init__(self, rank : float = 2200.):
        self.rank = rank
        
    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        first_term = (X_mat[:, 0]**2 + X_mat[:, 1] - 11.)**2
        second_term = (X_mat[:, 0] + X_mat[:, 1]**2 - 7.)**2
        X_eval = self.rank - (first_term + second_term)
        return ProblemUtils._to_evaluated(X, X_eval)


class BaseEggholder(IEvaluator):
    def __init__(self, rank : float = 1200.):
        self.rank = rank

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        first_term = - (X_mat[:, 1] + 47)*np.sin(np.sqrt(np.abs(X_mat[:, 0]/2. + (X_mat[:, 1] + 47))))
        second_term =  - X_mat[:, 0]*np.sin(np.sqrt(np.abs(X_mat[:, 0] - (X_mat[:, 1] + 47))))
        X_eval = self.rank - (first_term + second_term)
        return ProblemUtils._to_evaluated(X, X_eval)

class Sphere(IEvaluator):

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        X_eval = -np.sum(X_mat**2, axis = 1)
        return ProblemUtils._to_evaluated(X, X_eval)

class Schaffer(IEvaluator):

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix(X)
        X_eval = -418.9829*2 + np.sum(X_mat * np.sin(np.sqrt(np.abs(X_mat))), axis=1)
        return ProblemUtils._to_evaluated(X, X_eval)



from abc import abstractmethod
from hklearn_genetic.utils import ProblemUtils
from hklearn_genetic.Evaluators.IEvaluator import IEvaluator
from pymop.factory import get_problem
import numpy as np


class BaseRastrigin(IEvaluator):
    def __init__(self, rank : float = 100., dim = 2):
        self.rank = rank
        self.n_dim = dim

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix_phenotypes(X)
        X_eval = self.rank - (10.*self.n_dim + np.sum(X_mat**2 - 10.*np.cos(2.*np.pi*X_mat), axis = 1))
        return ProblemUtils._to_evaluated(X, X_eval)

class BaseBeale(IEvaluator):
    def __init__(self, rank : float = 150000.):
        self.rank = rank

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix_phenotypes(X)
        first_term = (1.5 - X_mat[:, 0] + X_mat[:, 0]*X_mat[:, 1])**2
        second_term = (2.25 - X_mat[:, 0] + X_mat[:, 0]*(X_mat[:, 1]**2))**2
        third_term = (2.625 - X_mat[:, 0] + X_mat[:, 0]*(X_mat[:, 1]**3))**2
        X_eval = self.rank - (first_term + second_term + third_term)
        return ProblemUtils._to_evaluated(X, X_eval)


class BaseHimmelblau(IEvaluator):
    def __init__(self, rank : float = 2200.):
        self.rank = rank
        
    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix_phenotypes(X)
        first_term = (X_mat[:, 0]**2 + X_mat[:, 1] - 11.)**2
        second_term = (X_mat[:, 0] + X_mat[:, 1]**2 - 7.)**2
        X_eval = self.rank - (first_term + second_term)
        return ProblemUtils._to_evaluated(X, X_eval)


class BaseEggholder(IEvaluator):
    def __init__(self, rank : float = 1200.):
        self.rank = rank

    def evaluate(self, X):
        X_mat = ProblemUtils._to_matrix_phenotypes(X)
        first_term = - (X_mat[:, 1] + 47)*np.sin(np.sqrt(np.abs(X_mat[:, 0]/2. + (X_mat[:, 1] + 47))))
        second_term =  - X_mat[:, 0]*np.sin(np.sqrt(np.abs(X_mat[:, 0] - (X_mat[:, 1] + 47))))
        X_eval = self.rank - (first_term + second_term)
        return ProblemUtils._to_evaluated(X, X_eval)

class BaseEvaluator(IEvaluator):
    def evaluate(self, X: list) -> list:
        X_mat = ProblemUtils._to_matrix_phenotypes(X)
        X_eval = self.evaluate_func(X_mat)
        # print(X_eval.shape)
        ProblemUtils._to_evaluated(X, X_eval)
    
    @abstractmethod
    def evaluate_func(self, X: np.array):
        pass

class Sphere(BaseEvaluator):
    def evaluate_func(self, X):
        return -np.sum(X**2, axis = 1)

class Schaffer(BaseEvaluator):
    def evaluate_func(self, X):
        return -418.9829*2 + np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

class DTLZ12D(BaseEvaluator):

    def __init__(self):
        self.ideal = np.array([0.,0.])
        self.nadir_point = np.array([1.,1.])
        self.dtlz1 = get_problem("dtlz1", n_var=12, n_obj=2)

    def evaluate_func(self, X: np.array) -> np.array:
        F = self.dtlz1.evaluate(X)
        f1, f2 = F[:,0], F[:, 1]
        f1 = f1.reshape((f1.shape[0], 1))
        f2 = f2.reshape((f2.shape[0], 1))
        f1 = (f1 - self.ideal[0])/(self.nadir_point[0] - self.ideal[0])
        f2 = (f2 - self.ideal[1])/(self.nadir_point[1] - self.ideal[1])
        F = np.concatenate((f1, f2), axis = 1)
        return F

    def g_dtlz1(self, X : np.array):
        a = 100.*(10 + ((X[:, 2:] - 0.5)**2 - np.cos(20*np.pi*(X[:, 2:] - 0.5))).sum(axis = 1))
        # print(a.shape)
        return a



class DTLZ1(BaseEvaluator):

    def __init__(self):
        self.ideal = np.array([0.,0.,0.])
        self.nadir_point = np.array([1.,1.,1.])
        self.dtlz1 = get_problem("dtlz1", n_var=12, n_obj=3)

    def evaluate_func(self, X: np.array) -> np.array:
        # print(X.shape)
        F = self.dtlz1.evaluate(X)
        f1, f2, f3 = F[:,0], F[:, 1], F[:, 2]
        # f1 = 0.5 * X[:, 0] * X[:, 1]*(1. + self.g_dtlz1(X))
        # f2 = 0.5 * X[:, 0] * (1 - X[:, 1])*(1. + self.g_dtlz1(X))
        # f3 = 0.5 * (1 - X[:, 0])*(1. + self.g_dtlz1(X))
        f1 = f1.reshape((f1.shape[0], 1))
        f2 = f2.reshape((f2.shape[0], 1))
        f3 = f3.reshape((f3.shape[0], 1))
        f1 = (f1 - self.ideal[0])/(self.nadir_point[0] - self.ideal[0])
        f2 = (f2 - self.ideal[1])/(self.nadir_point[1] - self.ideal[1])
        f3 = (f3 - self.ideal[2])/(self.nadir_point[2] - self.ideal[2])
        F = np.concatenate((f1, f2, f3), axis = 1)
        #print(fi)
        return F

    def g_dtlz1(self, X : np.array):
        a = 100.*(10 + ((X[:, 2:] - 0.5)**2 - np.cos(20*np.pi*(X[:, 2:] - 0.5))).sum(axis = 1))
        # print(a.shape)
        return a


class DTLZ2(BaseEvaluator):

    def __init__(self):
        self.ideal = np.array([0.,0.,0.])
        self.nadir_point = np.array([1.,1.,1.])

    def evaluate_func(self, X: np.array) -> np.array:
        # print(X.shape)
        f1 = np.cos(0.5*np.pi*X[:, 0])*np.cos(0.5*np.pi*X[:, 1])*(1 + self.g_dtlz2(X))
        f2 = np.cos(0.5*np.pi*X[:, 0])*np.sin(0.5*np.pi*X[:, 1])*(1 + self.g_dtlz2(X))
        f3 = np.sin(0.5*np.pi*X[:, 0])*(1 + self.g_dtlz2(X))
        f1 = f1.reshape((f1.shape[0], 1))
        f2 = f2.reshape((f2.shape[0], 1))
        f3 = f3.reshape((f3.shape[0], 1))
        f1 = (f1 - self.ideal[0])/(self.nadir_point[0] - self.ideal[0])
        f2 = (f2 - self.ideal[1])/(self.nadir_point[1] - self.ideal[1])
        f3 = (f3 - self.ideal[2])/(self.nadir_point[2] - self.ideal[2])
        F = np.concatenate((f1, f2, f3), axis = 1)
        #print(fi)
        return F

    def g_dtlz2(self, X : np.array):
        return ((X[:, 2:] - 0.5)**2).sum(axis = 1)


